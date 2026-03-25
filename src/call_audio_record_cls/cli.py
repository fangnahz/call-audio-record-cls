from __future__ import annotations

import argparse
import asyncio

from call_audio_record_cls.asr import AsrClient
from call_audio_record_cls.config import get_settings
from call_audio_record_cls.llm import LlmClient
from call_audio_record_cls.logging import configure_logging
from call_audio_record_cls.oss import OssClient
from call_audio_record_cls.pipeline import PipelineRunner


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the call audio ASR + classification pipeline.")
    parser.add_argument("--sample-limit", type=int, default=None, help="Process only N files.")
    return parser


async def run() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()

    settings = get_settings().model_copy(
        update={"sample_limit": args.sample_limit or get_settings().sample_limit}
    )
    runner = PipelineRunner(
        settings=settings,
        oss_client=OssClient(settings.oss, retry_settings=settings.retry),
        asr_client=AsrClient(
            settings=settings.asr,
            retry_settings=settings.retry,
            app_id=settings.asr_app_id,
            access_token=settings.asr_token,
        ),
        llm_client=LlmClient(
            settings=settings.llm,
            api_key=settings.llm_api_key,
            labels_path=settings.labels_path,
        ),
    )
    summary = await runner.run()
    print(summary.model_dump_json(indent=2))
    return 0 if not summary.failures else 1


def main() -> None:
    raise SystemExit(asyncio.run(run()))
