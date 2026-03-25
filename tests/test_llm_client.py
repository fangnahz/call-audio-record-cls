from __future__ import annotations

from pathlib import Path

from call_audio_record_cls.config import LlmSettings
from call_audio_record_cls.llm import LlmClient


class _FakeCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        self.kwargs = kwargs
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {
                                    "content": (
                                        '{"audio_file":"a.mp3","label":"高意向-主动询问车型",'
                                        '"confidence":0.92,"reason":"客户主动询问哈弗H6优惠",'
                                        '"transcript":"speaker 1: 你好\\n'
                                        'speaker 2: 我想问一下哈弗H6优惠",'
                                        '"intent_city":"合肥",'
                                        '"purchase_timeline":"下周",'
                                        '"intent_model":"哈弗H6"}'
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()


class _FakeChat:
    def __init__(self) -> None:
        self.completions = _FakeCompletions()


class _FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.chat = _FakeChat()


def test_format_transcript_preserves_order_and_speakers(tmp_path: Path) -> None:
    client = LlmClient(
        settings=LlmSettings(model="test-model"),
        api_key="token",
        labels_path=tmp_path / "labels.txt",
        async_client=_FakeAsyncOpenAI(),
    )
    raw_asr = {
        "result": {
            "utterances": [
                {"speaker": "2", "text": "您好，请问有什么可以帮您？", "start_time": 0.3},
                {"speaker": "1", "text": "我想问一下哈弗H6现在有什么优惠", "start_time": 0.1},
                {"speaker": "1", "text": "还有贷款方案吗", "start_time": 0.2},
            ]
        }
    }

    transcript = client.format_transcript(raw_asr)

    assert transcript == (
        "speaker 1: 我想问一下哈弗H6现在有什么优惠 还有贷款方案吗\n"
        "speaker 2: 您好，请问有什么可以帮您？"
    )


async def test_classify_returns_validated_result(tmp_path: Path) -> None:
    labels_path = tmp_path / "labels.txt"
    labels_path.write_text("• 示例标签：说明", encoding="utf-8")
    client = LlmClient(
        settings=LlmSettings(model="test-model"),
        api_key="token",
        labels_path=labels_path,
        async_client=_FakeAsyncOpenAI(),
    )
    raw_asr = {
        "utterances": [
            {"speaker": "A", "text": "你好", "start_time": 0},
            {"speaker": "B", "text": "我想问一下哈弗H6优惠", "start_time": 1},
        ]
    }

    result = await client.classify("oss://audio-1.mp3", raw_asr)

    assert result.oss_path == "oss://audio-1.mp3"
    assert result.label == "高意向-主动询问车型"
    assert result.confidence == 0.92
    assert "哈弗H6" in result.reason
    assert result.intent_city == "合肥"
    assert result.purchase_timeline == "下周"
    assert result.intent_model == "哈弗H6"


def test_parse_json_content_rejects_non_object(tmp_path: Path) -> None:
    client = LlmClient(
        settings=LlmSettings(model="test-model"),
        api_key="token",
        labels_path=tmp_path / "labels.txt",
        async_client=_FakeAsyncOpenAI(),
    )

    try:
        client._parse_json_content('["bad"]')
    except RuntimeError as exc:
        assert "JSON object" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for non-object JSON")


class _FallbackCompletions:
    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError(
                "Error code: 400 - {'error': {'message': 'response_format.type json_object is not supported by this model'}}"
            )
        return type(
            "Response",
            (),
            {
                "choices": [
                    type(
                        "Choice",
                        (),
                        {
                            "message": type(
                                "Message",
                                (),
                                {
                                    "content": (
                                        '{"audio_file":"a.mp3","label":"中意向-敷衍回复",'
                                        '"confidence":0.8,"reason":"fallback",'
                                        '"transcript":"speaker 1: hi",'
                                        '"intent_city":"",'
                                        '"purchase_timeline":"过几个月",'
                                        '"intent_model":"欧拉好猫"}'
                                    )
                                },
                            )()
                        },
                    )()
                ]
            },
        )()


class _FallbackChat:
    def __init__(self) -> None:
        self.completions = _FallbackCompletions()


class _FallbackAsyncOpenAI:
    def __init__(self) -> None:
        self.chat = _FallbackChat()


async def test_classify_falls_back_when_response_format_is_unsupported(tmp_path: Path) -> None:
    client_backend = _FallbackAsyncOpenAI()
    client = LlmClient(
        settings=LlmSettings(model="test-model"),
        api_key="token",
        labels_path=tmp_path / "labels.txt",
        async_client=client_backend,
    )

    result = await client.classify(
        "oss://audio-2.mp3",
        {"utterances": [{"speaker": "1", "text": "hello", "start_time": 0}]},
    )

    assert result.label == "中意向-敷衍回复"
    assert result.purchase_timeline == "过几个月"
    assert result.intent_model == "欧拉好猫"
    assert len(client_backend.chat.completions.calls) == 2
    assert "response_format" in client_backend.chat.completions.calls[0]
    assert "response_format" not in client_backend.chat.completions.calls[1]
