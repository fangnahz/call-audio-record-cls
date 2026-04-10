from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

_AUDIO_NAME_PATTERN = re.compile(r"(?P<name>audio-\d+)_(?P<rate>\d+)k_(?P<channels>[12])ch\.mp3$")


class AudioMetadata(BaseModel):
    sample_rate: int
    channels: int
    likely_role_separated: bool

    @classmethod
    def from_key(cls, key: str) -> "AudioMetadata":
        match = _AUDIO_NAME_PATTERN.search(key)
        if match is None:
            raise ValueError(f"Unsupported audio filename format: {key}")
        sample_rate = int(match.group("rate")) * 1000
        channels = int(match.group("channels"))
        return cls(
            sample_rate=sample_rate,
            channels=channels,
            likely_role_separated=channels == 2,
        )


class AudioObject(BaseModel):
    oss_key: str
    metadata: AudioMetadata

    @property
    def file_name(self) -> str:
        return self.oss_key.rsplit("/", 1)[-1]


class AsrJobRequest(BaseModel):
    audio: AudioObject
    signed_url: str


class AsrResultRecord(BaseModel):
    oss_path: str
    raw_result: dict[str, Any]


class SampleBatchRecord(BaseModel):
    oss_path: str
    asr_oss_path: str
    sample_rate: int
    channels: int
    status: str
    signed_url: str | None = None
    local_result_path: str | None = None
    error: str | None = None


class ClassificationResult(BaseModel):
    oss_path: str
    transcript: str
    label: str
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str
    intent_city: str = ""
    purchase_timeline: str = ""
    intent_model: str = ""
    llm_cost: str = ""


class PipelineFailure(BaseModel):
    oss_path: str
    stage: str
    error: str


class PipelineSummary(BaseModel):
    discovered: int = 0
    processed: int = 0
    skipped: int = 0
    failures: list[PipelineFailure] = Field(default_factory=list)
