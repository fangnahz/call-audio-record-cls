from __future__ import annotations

import httpx

from call_audio_record_cls.asr import AsrClient
from call_audio_record_cls.config import AsrSettings, RetrySettings, get_settings
from call_audio_record_cls.models import AsrJobRequest
from call_audio_record_cls.oss import OssClient


def _build_request() -> AsrJobRequest:
    settings = get_settings()
    audio = OssClient(settings.oss).parse_audio_object(
        "zhongqi-changcheng-recordings/audio-56_16k_2ch.mp3"
    )
    return AsrJobRequest(
        audio=audio,
        signed_url="https://signed.example/audio-56_16k_2ch.mp3",
    )


def test_submit_payload_uses_audio_metadata() -> None:
    client = AsrClient(
        settings=AsrSettings(),
        retry_settings=RetrySettings(),
        app_id="app-id",
        access_token="token",
    )

    payload = client._submit_payload(_build_request())

    assert payload["audio"]["url"] == "https://signed.example/audio-56_16k_2ch.mp3"
    assert payload["audio"]["rate"] == 16000
    assert payload["audio"]["channel"] == 2
    assert payload["request"]["enable_channel_split"] is True
    assert payload["request"]["enable_speaker_info"] is True


async def test_transcribe_polls_until_success() -> None:
    calls = {"count": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["count"] += 1
        if request.url.path.endswith("/submit"):
            return httpx.Response(
                200,
                headers={
                    "X-Api-Status-Code": "20000000",
                    "X-Api-Message": "submit ok",
                    "X-Tt-Logid": "logid-1",
                },
                json={},
            )
        if calls["count"] == 2:
            return httpx.Response(
                200,
                headers={
                    "X-Api-Status-Code": "20000001",
                    "X-Api-Message": "running",
                    "X-Tt-Logid": "logid-1",
                },
                json={},
            )
        return httpx.Response(
            200,
            headers={
                "X-Api-Status-Code": "20000000",
                "X-Api-Message": "finished",
                "X-Tt-Logid": "logid-1",
            },
            json={"result": {"utterances": []}},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = AsrClient(
        settings=AsrSettings(poll_interval_seconds=0.0, max_poll_attempts=5),
        retry_settings=RetrySettings(max_attempts=1),
        app_id="app-id",
        access_token="token",
        http_client=http_client,
    )

    result = await client.transcribe(_build_request())

    assert result == {"result": {"utterances": []}}
    assert calls["count"] == 3
    await http_client.aclose()


async def test_transcribe_raises_on_terminal_failure() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/submit"):
            return httpx.Response(
                200,
                headers={
                    "X-Api-Status-Code": "20000000",
                    "X-Api-Message": "submit ok",
                    "X-Tt-Logid": "logid-1",
                },
                json={},
            )
        return httpx.Response(
            400,
            headers={
                "X-Api-Status-Code": "40000001",
                "X-Api-Message": "bad request",
                "X-Tt-Logid": "logid-1",
            },
            json={},
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.AsyncClient(transport=transport)
    client = AsrClient(
        settings=AsrSettings(poll_interval_seconds=0.0, max_poll_attempts=2),
        retry_settings=RetrySettings(max_attempts=1),
        app_id="app-id",
        access_token="token",
        http_client=http_client,
    )

    try:
        await client.transcribe(_build_request())
    except RuntimeError as exc:
        assert "ASR query failed" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for terminal ASR failure")
    finally:
        await http_client.aclose()
