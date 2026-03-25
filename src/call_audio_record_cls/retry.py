from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import TypeVar

from tenacity import AsyncRetrying, retry_if_exception_type, stop_after_attempt, wait_exponential

from .config import RetrySettings

T = TypeVar("T")


class RetryableExternalError(RuntimeError):
    """Raised for transient external integration failures."""


async def run_with_retry(
    operation: Callable[[], Awaitable[T]],
    settings: RetrySettings,
) -> T:
    async for attempt in AsyncRetrying(
        stop=stop_after_attempt(settings.max_attempts),
        wait=wait_exponential(
            min=settings.min_wait_seconds,
            max=settings.max_wait_seconds,
        ),
        retry=retry_if_exception_type(RetryableExternalError),
        reraise=True,
    ):
        with attempt:
            return await operation()
    raise RuntimeError("Retry loop exited unexpectedly")
