from __future__ import annotations

import asyncio
import json
from pathlib import Path

from call_audio_record_cls.asr import AsrClient
from call_audio_record_cls.config import Settings
from call_audio_record_cls.logging import get_logger
from call_audio_record_cls.models import AsrJobRequest, AudioObject, SampleBatchRecord
from call_audio_record_cls.oss import OssClient

logger = get_logger(__name__)


class SmallBatchAsrHelper:
    def __init__(
        self,
        settings: Settings,
        oss_client: OssClient,
        asr_client: AsrClient,
    ) -> None:
        self.settings = settings
        self.oss_client = oss_client
        self.asr_client = asr_client
        self.oss_semaphore = asyncio.Semaphore(settings.concurrency.oss)
        self.asr_semaphore = asyncio.Semaphore(settings.concurrency.asr)

    async def preview(self, limit: int) -> list[SampleBatchRecord]:
        audio_files = await self._select_audio_files(limit)
        records: list[SampleBatchRecord] = []
        for audio in audio_files:
            asr_key = self.oss_client.build_asr_key(audio.oss_key)
            async with self.oss_semaphore:
                already_processed = await self.oss_client.object_exists(asr_key)
            records.append(
                SampleBatchRecord(
                    oss_path=audio.oss_key,
                    asr_oss_path=asr_key,
                    sample_rate=audio.metadata.sample_rate,
                    channels=audio.metadata.channels,
                    status="skip" if already_processed else "pending",
                )
            )
        await self._write_manifest(records, "preview.jsonl")
        return records

    async def run(self, limit: int) -> list[SampleBatchRecord]:
        audio_files = await self._select_audio_files(limit)
        records: list[SampleBatchRecord] = []
        tasks = [self._process_audio(audio, records) for audio in audio_files]
        if tasks:
            await asyncio.gather(*tasks)
        records.sort(key=lambda item: item.oss_path)
        await self._write_manifest(records, "run.jsonl")
        return records

    async def _select_audio_files(self, limit: int) -> list[AudioObject]:
        audio_files = await self.oss_client.list_audio_files()
        return audio_files[:limit]

    async def _process_audio(
        self,
        audio: AudioObject,
        records: list[SampleBatchRecord],
    ) -> None:
        asr_key = self.oss_client.build_asr_key(audio.oss_key)
        try:
            async with self.oss_semaphore:
                if await self.oss_client.object_exists(asr_key):
                    records.append(
                        SampleBatchRecord(
                            oss_path=audio.oss_key,
                            asr_oss_path=asr_key,
                            sample_rate=audio.metadata.sample_rate,
                            channels=audio.metadata.channels,
                            status="skip",
                        )
                    )
                    return
                signed_url = await self.oss_client.create_signed_url(audio.oss_key)

            async with self.asr_semaphore:
                raw_result = await self.asr_client.transcribe(
                    AsrJobRequest(audio=audio, signed_url=signed_url)
                )

            async with self.oss_semaphore:
                await self.oss_client.write_json(asr_key, raw_result)

            local_result_path = await self._write_local_asr_result(audio, raw_result)
            records.append(
                SampleBatchRecord(
                    oss_path=audio.oss_key,
                    asr_oss_path=asr_key,
                    sample_rate=audio.metadata.sample_rate,
                    channels=audio.metadata.channels,
                    status="processed",
                    signed_url=signed_url,
                    local_result_path=str(local_result_path),
                )
            )
            logger.info(
                "Processed ASR sample audio",
                extra={"structured": {"file_id": audio.oss_key, "stage": "asr-sample"}},
            )
        except Exception as exc:
            records.append(
                SampleBatchRecord(
                    oss_path=audio.oss_key,
                    asr_oss_path=asr_key,
                    sample_rate=audio.metadata.sample_rate,
                    channels=audio.metadata.channels,
                    status="failed",
                    error=str(exc),
                )
            )
            logger.exception(
                "ASR sample audio failed",
                extra={"structured": {"file_id": audio.oss_key, "error": str(exc)}},
            )

    async def _write_local_asr_result(self, audio: AudioObject, raw_result: dict) -> Path:
        output_dir = self._sample_output_dir() / "asr_results"
        output_dir.mkdir(parents=True, exist_ok=True)
        file_path = output_dir / f"{audio.file_name.removesuffix('.mp3')}.json"
        file_path.write_text(
            json.dumps(raw_result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return file_path

    async def _write_manifest(self, records: list[SampleBatchRecord], file_name: str) -> None:
        output_dir = self._sample_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / file_name
        manifest_path.write_text(
            "".join(record.model_dump_json() + "\n" for record in records),
            encoding="utf-8",
        )

    def _sample_output_dir(self) -> Path:
        return Path(self.settings.output_dir) / "sample_batch"
