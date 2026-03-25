from __future__ import annotations

from pathlib import Path

from call_audio_record_cls.config import LlmSettings, RetrySettings, get_settings
from call_audio_record_cls.models import ClassificationResult
from call_audio_record_cls.pipeline import PipelineRunner


class _RunnerFakeOssClient:
    def __init__(self, output_dir: Path) -> None:
        settings = get_settings().oss
        self.settings = settings
        self.output_dir = output_dir
        self._audio = [
            "zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3",
            "zhongqi-changcheng-recordings/audio-2_16k_2ch.mp3",
        ]
        self._existing = {
            "zhongqi-changcheng-asr/audio-1_48k_1ch.json",
            "zhongqi-changcheng-llm-v2/audio-2_16k_2ch.json",
        }
        self.written_json: dict[str, dict] = {}

    async def list_audio_files(self):
        from call_audio_record_cls.oss import OssClient

        client = OssClient(self.settings)
        return [client.parse_audio_object(key) for key in self._audio]

    async def object_exists(self, key: str) -> bool:
        return key in self._existing

    async def create_signed_url(self, key: str) -> str:
        return f"https://signed.example/{key}"

    async def read_json(self, key: str) -> dict:
        return {"utterances": [{"speaker": "1", "text": "existing asr", "start_time": 0}]}

    async def write_json(self, key: str, payload: dict) -> None:
        self.written_json[key] = payload

    async def write_jsonl(self, path: Path, rows: list[ClassificationResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(row.model_dump_json() for row in rows), encoding="utf-8")

    async def write_csv(self, path: Path, rows: list[ClassificationResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("header\n", encoding="utf-8")

    async def write_failure_jsonl(self, path: Path, failures: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(str(item) for item in failures), encoding="utf-8")

    def build_asr_key(self, audio_key: str) -> str:
        return audio_key.replace("zhongqi-changcheng-recordings/", "zhongqi-changcheng-asr/").replace(
            ".mp3", ".json"
        )

    def build_llm_key(self, audio_key: str) -> str:
        return audio_key.replace(
            "zhongqi-changcheng-recordings/",
            "zhongqi-changcheng-llm-v2/",
        ).replace(".mp3", ".json")

    def parse_audio_object(self, key: str):
        from call_audio_record_cls.oss import OssClient

        return OssClient(self.settings).parse_audio_object(key)


class _RunnerFakeAsrClient:
    def __init__(self) -> None:
        self.calls = 0

    async def transcribe(self, request):
        self.calls += 1
        return {"utterances": [{"speaker": "1", "text": "fresh asr", "start_time": 0}]}


class _RunnerFakeLlmClient:
    def __init__(self) -> None:
        self.calls = 0

    async def classify(self, oss_path: str, raw_asr_result: dict) -> ClassificationResult:
        self.calls += 1
        transcript = raw_asr_result["utterances"][0]["text"]
        return ClassificationResult(
            oss_path=oss_path,
            transcript=f"speaker 1: {transcript}",
            label="中意向-敷衍回复",
            confidence=0.6,
            reason="test",
            intent_city="合肥",
            purchase_timeline="下周",
            intent_model="哈弗H6",
        )


async def test_runner_reuses_existing_asr_and_skips_existing_llm(tmp_path: Path) -> None:
    settings = get_settings().model_copy(
        update={
            "output_dir": tmp_path,
            "retry": RetrySettings(max_attempts=1),
            "llm": LlmSettings(model="test-model"),
        }
    )
    oss_client = _RunnerFakeOssClient(tmp_path)
    asr_client = _RunnerFakeAsrClient()
    llm_client = _RunnerFakeLlmClient()

    summary = await PipelineRunner(
        settings=settings,
        oss_client=oss_client,
        asr_client=asr_client,
        llm_client=llm_client,
    ).run()

    assert summary.discovered == 2
    assert summary.processed == 1
    assert summary.skipped == 1
    assert asr_client.calls == 0
    assert llm_client.calls == 1
    assert "zhongqi-changcheng-llm-v2/audio-1_48k_1ch.json" in oss_client.written_json
