from types import SimpleNamespace

from call_audio_record_cls.config import get_settings
from call_audio_record_cls.oss import OssClient


def test_settings_load() -> None:
    settings = get_settings()
    assert settings.oss.bucket == "yiwise-asr"


def test_oss_key_mapping() -> None:
    client = OssClient(get_settings().oss)
    source = "zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3"
    assert client.build_asr_key(source) == "zhongqi-changcheng-asr/audio-1_48k_1ch.json"
    assert client.build_llm_key(source) == "zhongqi-changcheng-llm-v3/audio-1_48k_1ch.json"


class _FakePage:
    def __init__(self, keys: list[str]) -> None:
        self.contents = [SimpleNamespace(key=key) for key in keys]


class _FakePaginator:
    def __init__(self, keys: list[str]) -> None:
        self._keys = keys

    def iter_page(self, request: object):
        assert getattr(request, "bucket") == "yiwise-asr"
        assert getattr(request, "prefix") == "zhongqi-changcheng-recordings/"
        yield _FakePage(self._keys)


class _FakeSdkClient:
    def __init__(self) -> None:
        self.presign_calls: list[object] = []
        self.get_object_calls: list[object] = []

    def list_objects_v2_paginator(self) -> _FakePaginator:
        return _FakePaginator(
            [
                "zhongqi-changcheng-recordings/audio-2_16k_2ch.mp3",
                "zhongqi-changcheng-recordings/readme.txt",
                "zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3",
            ]
        )

    def presign(self, request: object):
        self.presign_calls.append(request)
        return SimpleNamespace(url=f"https://signed.example/{getattr(request, 'key')}")

    def get_object(self, request: object):
        self.get_object_calls.append(request)
        return SimpleNamespace(body=SimpleNamespace(read=lambda: b'{"ok": true}'))


class _FakeRequest:
    def __init__(self, **kwargs) -> None:
        for key, value in kwargs.items():
            setattr(self, key, value)


class _FakeSdkModule:
    ListObjectsV2Request = _FakeRequest
    GetObjectRequest = _FakeRequest


async def test_list_audio_files_filters_and_sorts() -> None:
    client = OssClient(
        get_settings().oss,
        sdk_client=_FakeSdkClient(),
        sdk_module=_FakeSdkModule(),
    )

    audio_files = await client.list_audio_files()

    assert [audio.oss_key for audio in audio_files] == [
        "zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3",
        "zhongqi-changcheng-recordings/audio-2_16k_2ch.mp3",
    ]
    assert audio_files[0].metadata.sample_rate == 48000
    assert audio_files[1].metadata.channels == 2


async def test_create_signed_url_uses_object_key() -> None:
    fake_sdk_client = _FakeSdkClient()
    client = OssClient(
        get_settings().oss,
        sdk_client=fake_sdk_client,
        sdk_module=_FakeSdkModule(),
    )

    signed_url = await client.create_signed_url(
        "zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3"
    )

    assert signed_url.endswith("zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3")
    assert getattr(fake_sdk_client.presign_calls[0], "bucket") == "yiwise-asr"


async def test_read_json_uses_object_key() -> None:
    fake_sdk_client = _FakeSdkClient()
    client = OssClient(
        get_settings().oss,
        sdk_client=fake_sdk_client,
        sdk_module=_FakeSdkModule(),
    )

    payload = await client.read_json("zhongqi-changcheng-asr/audio-1_48k_1ch.json")

    assert payload == {"ok": True}
    assert getattr(fake_sdk_client.get_object_calls[0], "key") == (
        "zhongqi-changcheng-asr/audio-1_48k_1ch.json"
    )


def test_not_found_detection_handles_sdk_message_strings() -> None:
    exc = RuntimeError(
        "operation error HeadObject: Error returned by Service.\n"
        "Http Status Code: 404.\n"
        "Error Code: NoSuchKey.\n"
    )

    assert OssClient._is_not_found_error(exc) is True
