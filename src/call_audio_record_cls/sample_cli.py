from __future__ import annotations

import argparse
import asyncio

from call_audio_record_cls.asr import AsrClient
from call_audio_record_cls.config import get_settings
from call_audio_record_cls.logging import configure_logging
from call_audio_record_cls.oss import OssClient
from call_audio_record_cls.pipeline import SmallBatchAsrHelper


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview or run a small ASR sample batch.")
    parser.add_argument(
        "command",
        choices=["preview", "run"],
        help="Preview sample candidates or run ASR for the sample set.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=5,
        help="Number of audio files to include in the small batch.",
    )
    return parser


async def run() -> int:
    parser = build_parser()
    args = parser.parse_args()
    configure_logging()

    settings = get_settings()
    helper = SmallBatchAsrHelper(
        settings=settings,
        oss_client=OssClient(settings.oss, retry_settings=settings.retry),
        asr_client=AsrClient(
            settings=settings.asr,
            retry_settings=settings.retry,
            app_id=settings.asr_app_id,
            access_token=settings.asr_token,
        ),
    )

    if args.command == "preview":
        records = await helper.preview(args.limit)
    else:
        records = await helper.run(args.limit)

    print("\n".join(record.model_dump_json() for record in records))
    return 0 if all(record.status != "failed" for record in records) else 1


def main() -> None:
    raise SystemExit(asyncio.run(run()))
