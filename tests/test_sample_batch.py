from __future__ import annotations

from pathlib import Path

from call_audio_record_cls.config import get_settings
from call_audio_record_cls.oss import OssClient
from call_audio_record_cls.pipeline import SmallBatchAsrHelper


class _FakeAsrClient:
    async def transcribe(self, request):
        return {
            "audio_url": request.signed_url,
            "sample_rate": request.audio.metadata.sample_rate,
            "channels": request.audio.metadata.channels,
        }


class _FakeOssClient:
    def __init__(self, settings) -> None:
        self._client = OssClient(settings)
        self._existing: set[str] = set()
        self.writes: dict[str, dict] = {}

    async def list_audio_files(self):
        return [
            self._client.parse_audio_object("zhongqi-changcheng-recordings/audio-1_48k_1ch.mp3"),
            self._client.parse_audio_object("zhongqi-changcheng-recordings/audio-2_16k_2ch.mp3"),
        ]

    async def object_exists(self, key: str) -> bool:
        return key in self._existing

    async def create_signed_url(self, key: str) -> str:
        return f"https://signed.example/{key}"

    async def write_json(self, key: str, payload: dict) -> None:
        self.writes[key] = payload

    def build_asr_key(self, audio_key: str) -> str:
        return self._client.build_asr_key(audio_key)


async def test_sample_preview_marks_existing_asr_outputs(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"output_dir": tmp_path})
    fake_oss = _FakeOssClient(settings.oss)
    fake_oss._existing.add("zhongqi-changcheng-asr/audio-1_48k_1ch.json")
    helper = SmallBatchAsrHelper(settings=settings, oss_client=fake_oss, asr_client=_FakeAsrClient())

    records = await helper.preview(2)

    assert [record.status for record in records] == ["skip", "pending"]
    manifest = (tmp_path / "sample_batch" / "preview.jsonl").read_text(encoding="utf-8")
    assert '"status":"skip"' in manifest


async def test_sample_run_processes_pending_audio(tmp_path: Path) -> None:
    settings = get_settings().model_copy(update={"output_dir": tmp_path})
    fake_oss = _FakeOssClient(settings.oss)
    fake_oss._existing.add("zhongqi-changcheng-asr/audio-1_48k_1ch.json")
    helper = SmallBatchAsrHelper(settings=settings, oss_client=fake_oss, asr_client=_FakeAsrClient())

    records = await helper.run(2)

    assert [record.status for record in records] == ["skip", "processed"]
    assert "zhongqi-changcheng-asr/audio-2_16k_2ch.json" in fake_oss.writes
    local_file = tmp_path / "sample_batch" / "asr_results" / "audio-2_16k_2ch.json"
    assert local_file.exists()
