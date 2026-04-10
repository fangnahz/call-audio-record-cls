from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from call_audio_record_cls.config import OssSettings, RetrySettings
from call_audio_record_cls.models import AudioMetadata, AudioObject, ClassificationResult
from call_audio_record_cls.retry import RetryableExternalError, run_with_retry


class OssClient:
    """Alibaba OSS abstraction for object iteration, existence checks, signed URLs, and writes."""

    def __init__(
        self,
        settings: OssSettings,
        retry_settings: RetrySettings | None = None,
        sdk_client: Any | None = None,
        sdk_module: Any | None = None,
    ) -> None:
        self.settings = settings
        self.retry_settings = retry_settings
        self._sdk_client = sdk_client
        self._sdk_module = sdk_module

    async def list_audio_files(self) -> list[AudioObject]:
        def operation() -> list[AudioObject]:
            client = self._get_sdk_client()
            oss = self._get_sdk_module()
            paginator = client.list_objects_v2_paginator()
            request = oss.ListObjectsV2Request(
                bucket=self.settings.bucket,
                prefix=self.settings.input_prefix,
            )
            audio_files: list[AudioObject] = []
            for page in paginator.iter_page(request):
                for obj in getattr(page, "contents", []):
                    key = getattr(obj, "key", "")
                    if not key.endswith(".mp3"):
                        continue
                    audio_files.append(self.parse_audio_object(key))
            audio_files.sort(key=lambda item: item.oss_key)
            return audio_files

        return await self._run_sdk_operation(operation)

    async def object_exists(self, key: str) -> bool:
        def operation() -> bool:
            client = self._get_sdk_client()
            oss = self._get_sdk_module()
            try:
                client.head_object(
                    oss.HeadObjectRequest(
                        bucket=self.settings.bucket,
                        key=key,
                    )
                )
                return True
            except Exception as exc:
                if self._is_not_found_error(exc):
                    return False
                raise self._normalize_sdk_exception(exc) from exc

        return await self._run_sdk_operation(operation)

    async def create_signed_url(self, key: str) -> str:
        def operation() -> str:
            client = self._get_sdk_client()
            oss = self._get_sdk_module()
            presigned = client.presign(
                oss.GetObjectRequest(
                    bucket=self.settings.bucket,
                    key=key,
                )
            )
            return presigned.url

        return await self._run_sdk_operation(operation)

    async def read_json(self, key: str) -> dict:
        def operation() -> dict:
            client = self._get_sdk_client()
            oss = self._get_sdk_module()
            result = client.get_object(
                oss.GetObjectRequest(
                    bucket=self.settings.bucket,
                    key=key,
                )
            )
            body = result.body.read()
            if isinstance(body, bytes):
                text = body.decode("utf-8")
            else:
                text = str(body)
            return json.loads(text)

        return await self._run_sdk_operation(operation)

    async def write_json(self, key: str, payload: dict) -> None:
        def operation() -> None:
            client = self._get_sdk_client()
            oss = self._get_sdk_module()
            body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            client.put_object(
                oss.PutObjectRequest(
                    bucket=self.settings.bucket,
                    key=key,
                    body=body,
                )
            )

        await self._run_sdk_operation(operation)

    async def write_jsonl(self, path: Path, rows: list[ClassificationResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = "\n".join(row.model_dump_json() for row in rows)
        path.write_text(f"{serialized}\n" if serialized else "", encoding="utf-8")

    async def write_failure_jsonl(self, path: Path, failures: list[dict[str, str]]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        serialized = "\n".join(json.dumps(row, ensure_ascii=False) for row in failures)
        path.write_text(f"{serialized}\n" if serialized else "", encoding="utf-8")

    async def write_csv(self, path: Path, rows: list[ClassificationResult]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "oss_path,transcript,label,confidence,reason,intent_city,purchase_timeline,intent_model,llm_cost"
        ]
        for row in rows:
            values = [
                json.dumps(row.oss_path, ensure_ascii=False),
                json.dumps(row.transcript, ensure_ascii=False),
                json.dumps(row.label, ensure_ascii=False),
                str(row.confidence),
                json.dumps(row.reason, ensure_ascii=False),
                json.dumps(row.intent_city, ensure_ascii=False),
                json.dumps(row.purchase_timeline, ensure_ascii=False),
                json.dumps(row.intent_model, ensure_ascii=False),
                json.dumps(row.llm_cost, ensure_ascii=False),
            ]
            lines.append(",".join(values))
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def build_asr_key(self, audio_key: str) -> str:
        return audio_key.replace(
            self.settings.input_prefix,
            self.settings.asr_prefix,
            1,
        ).removesuffix(".mp3") + ".json"

    def build_llm_key(self, audio_key: str) -> str:
        return audio_key.replace(
            self.settings.input_prefix,
            self.settings.llm_prefix,
            1,
        ).removesuffix(".mp3") + ".json"

    @staticmethod
    def parse_audio_object(key: str) -> AudioObject:
        return AudioObject(oss_key=key, metadata=AudioMetadata.from_key(key))

    async def _run_sdk_operation(self, operation: Any) -> Any:
        async def wrapped() -> Any:
            try:
                return await asyncio.to_thread(operation)
            except Exception as exc:
                if isinstance(exc, (RetryableExternalError, RuntimeError)):
                    raise
                raise self._normalize_sdk_exception(exc) from exc

        if self.retry_settings is None:
            return await wrapped()
        return await run_with_retry(wrapped, self.retry_settings)

    def _get_sdk_module(self) -> Any:
        if self._sdk_module is None:
            try:
                import alibabacloud_oss_v2 as oss
            except ImportError as exc:
                raise RuntimeError(
                    "Alibaba OSS SDK is not installed. Run `uv sync` to install project dependencies."
                ) from exc
            self._sdk_module = oss
        return self._sdk_module

    def _get_sdk_client(self) -> Any:
        if self._sdk_client is None:
            oss = self._get_sdk_module()
            credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
            cfg = oss.config.load_default()
            cfg.credentials_provider = credentials_provider
            cfg.region = self.settings.region
            cfg.endpoint = self.settings.endpoint
            self._sdk_client = oss.Client(cfg)
        return self._sdk_client

    @staticmethod
    def _extract_status_code(exc: Exception) -> int | None:
        raw = getattr(exc, "status_code", None) or getattr(exc, "http_status_code", None)
        if raw is None:
            return None
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _extract_error_code(exc: Exception) -> str | None:
        code = getattr(exc, "code", None) or getattr(exc, "error_code", None)
        return str(code) if code is not None else None

    @classmethod
    def _is_not_found_error(cls, exc: Exception) -> bool:
        status_code = cls._extract_status_code(exc)
        error_code = cls._extract_error_code(exc)
        message = str(exc)
        return (
            status_code == 404
            or error_code in {"NoSuchKey", "NotFound", "NoSuchBucket"}
            or "Http Status Code: 404" in message
            or "Error Code: NoSuchKey" in message
            or "Error Code: NotFound" in message
        )

    @classmethod
    def _normalize_sdk_exception(cls, exc: Exception) -> Exception:
        status_code = cls._extract_status_code(exc)
        if status_code is not None and 400 <= status_code < 500 and status_code != 429:
            return RuntimeError(str(exc))
        return RetryableExternalError(str(exc))
