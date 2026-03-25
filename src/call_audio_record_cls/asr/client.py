from __future__ import annotations

import asyncio
import uuid

import httpx

from call_audio_record_cls.config import AsrSettings, RetrySettings
from call_audio_record_cls.models import AsrJobRequest
from call_audio_record_cls.retry import RetryableExternalError, run_with_retry

_STATUS_SUCCESS = "20000000"
_STATUS_RUNNING = {"20000001", "20000002"}


class AsrClient:
    """ASR integration against the submit/query HTTP API."""

    def __init__(
        self,
        settings: AsrSettings,
        retry_settings: RetrySettings,
        app_id: str | None,
        access_token: str | None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.settings = settings
        self.retry_settings = retry_settings
        self.app_id = app_id
        self.access_token = access_token
        self._http_client = http_client

    async def transcribe(self, request: AsrJobRequest) -> dict:
        if not self.app_id or not self.access_token:
            raise RuntimeError("Missing ASR credentials: set ASR_APP_ID and ASR_TOKEN in .env")

        async def operation() -> dict:
            task_id, x_tt_logid = await self._submit_task(request)
            return await self._poll_task(task_id=task_id, x_tt_logid=x_tt_logid)

        return await run_with_retry(operation, self.retry_settings)

    async def _submit_task(self, request: AsrJobRequest) -> tuple[str, str]:
        task_id = str(uuid.uuid4())
        response = await self._post_json(
            url=self.settings.submit_url,
            headers=self._submit_headers(task_id),
            payload=self._submit_payload(request),
        )
        status_code = self._status_code(response)
        if status_code != _STATUS_SUCCESS:
            self._raise_for_status(response, stage="submit")
        x_tt_logid = response.headers.get("X-Tt-Logid", "")
        return task_id, x_tt_logid

    async def _poll_task(self, task_id: str, x_tt_logid: str) -> dict:
        for _ in range(self.settings.max_poll_attempts):
            response = await self._post_json(
                url=self.settings.query_url,
                headers=self._query_headers(task_id, x_tt_logid),
                payload={},
            )
            status_code = self._status_code(response)
            if status_code == _STATUS_SUCCESS:
                return response.json()
            if status_code in _STATUS_RUNNING:
                await asyncio.sleep(self.settings.poll_interval_seconds)
                continue
            self._raise_for_status(response, stage="query")
        raise RetryableExternalError("ASR polling exceeded the maximum number of attempts")

    def _submit_headers(self, task_id: str) -> dict[str, str]:
        return {
            "X-Api-App-Key": self.app_id or "",
            "X-Api-Access-Key": self.access_token or "",
            "X-Api-Resource-Id": self.settings.resource_id,
            "X-Api-Request-Id": task_id,
            "X-Api-Sequence": "-1",
        }

    def _query_headers(self, task_id: str, x_tt_logid: str) -> dict[str, str]:
        headers = {
            "X-Api-App-Key": self.app_id or "",
            "X-Api-Access-Key": self.access_token or "",
            "X-Api-Resource-Id": self.settings.resource_id,
            "X-Api-Request-Id": task_id,
        }
        if x_tt_logid:
            headers["X-Tt-Logid"] = x_tt_logid
        return headers

    def _submit_payload(self, request: AsrJobRequest) -> dict:
        audio_payload = {
            "url": request.signed_url,
            "format": "mp3",
            "rate": request.audio.metadata.sample_rate,
            "channel": request.audio.metadata.channels,
        }
        request_payload = {
            "model_name": self.settings.model_name,
            "enable_channel_split": self.settings.enable_channel_split,
            "enable_ddc": self.settings.enable_ddc,
            "enable_speaker_info": self.settings.enable_speaker_info,
            "enable_punc": self.settings.enable_punc,
            "enable_itn": self.settings.enable_itn,
            "corpus": {
                "correct_table_name": "",
                "context": "",
            },
        }
        if self.settings.language:
            request_payload["language"] = self.settings.language
        return {
            "user": {"uid": self.settings.user_id},
            "audio": audio_payload,
            "request": request_payload,
        }

    async def _post_json(
        self,
        url: str,
        headers: dict[str, str],
        payload: dict,
    ) -> httpx.Response:
        client = self._get_http_client()
        try:
            response = await client.post(url, headers=headers, json=payload)
            return response
        except httpx.TimeoutException as exc:
            raise RetryableExternalError(f"ASR request timed out: {exc}") from exc
        except httpx.NetworkError as exc:
            raise RetryableExternalError(f"ASR network error: {exc}") from exc

    def _get_http_client(self) -> httpx.AsyncClient:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(timeout=self.settings.request_timeout_seconds)
        return self._http_client

    @staticmethod
    def _status_code(response: httpx.Response) -> str:
        return response.headers.get("X-Api-Status-Code", "")

    def _raise_for_status(self, response: httpx.Response, stage: str) -> None:
        status_code = self._status_code(response)
        message = response.headers.get("X-Api-Message", "")
        detail = f"ASR {stage} failed: status_code={status_code or 'missing'} message={message}"
        if response.status_code >= 500 or status_code == "429":
            raise RetryableExternalError(detail)
        if response.status_code == 429:
            raise RetryableExternalError(detail)
        raise RuntimeError(detail)
