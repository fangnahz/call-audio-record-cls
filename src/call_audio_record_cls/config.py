from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()


class OssSettings(BaseModel):
    bucket: str = "yiwise-asr"
    input_prefix: str = "zhongqi-changcheng-recordings/"
    asr_prefix: str = "zhongqi-changcheng-asr/"
    llm_prefix: str = "zhongqi-changcheng-llm-v2/"
    region: str = "cn-hangzhou"
    endpoint: str = "oss-cn-hangzhou.aliyuncs.com"
    signed_url_ttl_seconds: int = 900


class ConcurrencySettings(BaseModel):
    oss: int = 10
    asr: int = 10
    llm: int = 20


class RetrySettings(BaseModel):
    max_attempts: int = 3
    min_wait_seconds: float = 1.0
    max_wait_seconds: float = 8.0


class AsrSettings(BaseModel):
    submit_url: str = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/submit"
    query_url: str = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query"
    resource_id: str = "volc.bigasr.auc"
    model_name: str = "bigmodel"
    language: str | None = None
    poll_interval_seconds: float = 1.0
    max_poll_attempts: int = 300
    request_timeout_seconds: float = 30.0
    enable_channel_split: bool = True
    enable_ddc: bool = True
    enable_speaker_info: bool = True
    enable_punc: bool = True
    enable_itn: bool = True
    user_id: str = "fake_uid"


class LlmSettings(BaseModel):
    base_url: str | None = None
    model: str = "TODO-model-id"
    request_timeout_seconds: float = 60.0
    use_response_format_json: bool = True


class Settings(BaseModel):
    env_file: Path = Path(".env")
    output_dir: Path = Field(default_factory=lambda: Path(os.getenv("OUTPUT_DIR", "output")))
    labels_path: Path = Field(default_factory=lambda: Path(os.getenv("LABELS_PATH", "labels.txt")))
    sample_limit: int | None = Field(default=None, ge=1)
    asr: AsrSettings = Field(default_factory=AsrSettings)
    llm: LlmSettings = Field(default_factory=LlmSettings)
    oss: OssSettings = Field(default_factory=OssSettings)
    concurrency: ConcurrencySettings = Field(default_factory=ConcurrencySettings)
    retry: RetrySettings = Field(default_factory=RetrySettings)
    asr_app_id: str | None = Field(default_factory=lambda: os.getenv("ASR_APP_ID"))
    asr_token: str | None = Field(default_factory=lambda: os.getenv("ASR_TOKEN"))
    llm_api_key: str | None = Field(default_factory=lambda: os.getenv("LLM_API_KEY"))


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    asr = AsrSettings(
        submit_url=os.getenv(
            "ASR_SUBMIT_URL",
            "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/submit",
        ),
        query_url=os.getenv(
            "ASR_QUERY_URL",
            "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query",
        ),
        resource_id=os.getenv("ASR_RESOURCE_ID", "volc.bigasr.auc"),
        model_name=os.getenv("ASR_MODEL_NAME", "bigmodel"),
        language=os.getenv("ASR_LANGUAGE"),
        poll_interval_seconds=float(os.getenv("ASR_POLL_INTERVAL_SECONDS", "1.0")),
        max_poll_attempts=int(os.getenv("ASR_MAX_POLL_ATTEMPTS", "300")),
        request_timeout_seconds=float(os.getenv("ASR_REQUEST_TIMEOUT_SECONDS", "30.0")),
        enable_channel_split=os.getenv("ASR_ENABLE_CHANNEL_SPLIT", "true").lower() == "true",
        enable_ddc=os.getenv("ASR_ENABLE_DDC", "true").lower() == "true",
        enable_speaker_info=os.getenv("ASR_ENABLE_SPEAKER_INFO", "true").lower() == "true",
        enable_punc=os.getenv("ASR_ENABLE_PUNC", "true").lower() == "true",
        enable_itn=os.getenv("ASR_ENABLE_ITN", "true").lower() == "true",
        user_id=os.getenv("ASR_USER_ID", "fake_uid"),
    )
    oss = OssSettings(
        bucket=os.getenv("OSS_BUCKET", "yiwise-asr"),
        input_prefix=os.getenv("OSS_INPUT_PREFIX", "zhongqi-changcheng-recordings/"),
        asr_prefix=os.getenv("OSS_ASR_PREFIX", "zhongqi-changcheng-asr/"),
        llm_prefix=os.getenv("OSS_LLM_PREFIX", "zhongqi-changcheng-llm-v2/"),
        region=os.getenv("OSS_REGION", "cn-hangzhou"),
        endpoint=os.getenv("OSS_ENDPOINT", "oss-cn-hangzhou.aliyuncs.com"),
        signed_url_ttl_seconds=int(os.getenv("OSS_SIGNED_URL_TTL_SECONDS", "900")),
    )
    concurrency = ConcurrencySettings(
        oss=int(os.getenv("OSS_CONCURRENCY", "10")),
        asr=int(os.getenv("ASR_CONCURRENCY", "10")),
        llm=int(os.getenv("LLM_CONCURRENCY", "5")),
    )
    retry = RetrySettings(
        max_attempts=int(os.getenv("MAX_RETRY_ATTEMPTS", "3")),
        min_wait_seconds=float(os.getenv("MIN_RETRY_WAIT_SECONDS", "1.0")),
        max_wait_seconds=float(os.getenv("MAX_RETRY_WAIT_SECONDS", "8.0")),
    )
    llm = LlmSettings(
        base_url=os.getenv("LLM_BASE_URL"),
        model=os.getenv("LLM_MODEL", "TODO-model-id"),
        request_timeout_seconds=float(os.getenv("LLM_REQUEST_TIMEOUT_SECONDS", "60.0")),
        use_response_format_json=os.getenv("LLM_USE_RESPONSE_FORMAT_JSON", "true").lower()
        == "true",
    )
    sample_limit = os.getenv("SAMPLE_LIMIT")
    return Settings(
        asr=asr,
        llm=llm,
        oss=oss,
        concurrency=concurrency,
        retry=retry,
        sample_limit=int(sample_limit) if sample_limit else None,
    )
