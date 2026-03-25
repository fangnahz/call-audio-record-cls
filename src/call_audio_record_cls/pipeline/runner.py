from __future__ import annotations

import asyncio
import time
from pathlib import Path

from call_audio_record_cls.asr import AsrClient
from call_audio_record_cls.config import Settings
from call_audio_record_cls.llm import LlmClient
from call_audio_record_cls.logging import get_logger
from call_audio_record_cls.models import (
    AsrJobRequest,
    ClassificationResult,
    PipelineFailure,
    PipelineSummary,
)
from call_audio_record_cls.oss import OssClient
from call_audio_record_cls.retry import run_with_retry

logger = get_logger(__name__)


class PipelineRunner:
    def __init__(
        self,
        settings: Settings,
        oss_client: OssClient,
        asr_client: AsrClient,
        llm_client: LlmClient,
    ) -> None:
        self.settings = settings
        self.oss_client = oss_client
        self.asr_client = asr_client
        self.llm_client = llm_client
        self.oss_semaphore = asyncio.Semaphore(settings.concurrency.oss)
        self.asr_semaphore = asyncio.Semaphore(settings.concurrency.asr)
        self.llm_semaphore = asyncio.Semaphore(settings.concurrency.llm)

    async def run(self) -> PipelineSummary:
        summary = PipelineSummary()
        results: list[ClassificationResult] = []
        audio_files = await self.oss_client.list_audio_files()
        if self.settings.sample_limit is not None:
            audio_files = audio_files[: self.settings.sample_limit]
        summary.discovered = len(audio_files)

        tasks = [self._process_audio(audio.oss_key, results, summary) for audio in audio_files]
        if tasks:
            await asyncio.gather(*tasks)

        await self._write_outputs(results, summary)
        return summary

    async def _process_audio(
        self,
        audio_key: str,
        results: list[ClassificationResult],
        summary: PipelineSummary,
    ) -> None:
        started_at = time.perf_counter()
        asr_key = self.oss_client.build_asr_key(audio_key)
        llm_key = self.oss_client.build_llm_key(audio_key)

        try:
            async with self.oss_semaphore:
                if await self.oss_client.object_exists(llm_key):
                    summary.skipped += 1
                    logger.info(
                        "Skipping already processed audio",
                        extra={"structured": {"file_id": audio_key, "stage": "skip"}},
                    )
                    return

                audio = self.oss_client.parse_audio_object(audio_key)
                asr_exists = await self.oss_client.object_exists(asr_key)

            if asr_exists:
                async with self.oss_semaphore:
                    raw_asr_result = await self.oss_client.read_json(asr_key)
            else:
                async with self.oss_semaphore:
                    signed_url = await self.oss_client.create_signed_url(audio_key)

                async with self.asr_semaphore:
                    raw_asr_result = await self.asr_client.transcribe(
                        AsrJobRequest(audio=audio, signed_url=signed_url)
                    )

                async with self.oss_semaphore:
                    await self.oss_client.write_json(asr_key, raw_asr_result)

            async with self.llm_semaphore:
                classification = await run_with_retry(
                    lambda: self.llm_client.classify(audio_key, raw_asr_result),
                    self.settings.retry,
                )

            async with self.oss_semaphore:
                await self.oss_client.write_json(llm_key, classification.model_dump())

            results.append(classification)
            summary.processed += 1
            logger.info(
                "Processed audio",
                extra={
                    "structured": {
                        "file_id": audio_key,
                        "asr_source": "cache" if asr_exists else "fresh",
                        "label": classification.label,
                        "latency_seconds": round(time.perf_counter() - started_at, 3),
                    }
                },
            )
        except Exception as exc:
            summary.failures.append(
                PipelineFailure(
                    oss_path=audio_key,
                    stage=self._infer_failure_stage(exc, asr_exists if "asr_exists" in locals() else False),
                    error=str(exc),
                )
            )
            logger.exception(
                "Audio processing failed",
                extra={"structured": {"file_id": audio_key, "error": str(exc)}},
            )

    async def _write_outputs(
        self,
        rows: list[ClassificationResult],
        summary: PipelineSummary,
    ) -> None:
        output_dir = Path(self.settings.output_dir)
        await self.oss_client.write_jsonl(output_dir / "classification_results.jsonl", rows)
        await self.oss_client.write_csv(output_dir / "classification_results.csv", rows)
        await self.oss_client.write_failure_jsonl(
            output_dir / "pipeline_failures.jsonl",
            [failure.model_dump() for failure in summary.failures],
        )

    @staticmethod
    def _infer_failure_stage(exc: Exception, asr_loaded: bool) -> str:
        message = str(exc).lower()
        if "llm" in message or "json object" in message or "transcript" in message:
            return "llm"
        if "asr" in message or not asr_loaded:
            return "asr"
        return "pipeline"
