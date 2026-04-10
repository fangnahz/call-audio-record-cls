from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI

from call_audio_record_cls.config import LlmSettings
from call_audio_record_cls.models import ClassificationResult
from call_audio_record_cls.retry import RetryableExternalError

DEFAULT_LABEL_SET = """
• 高意向-主动询问车型： 客户在对话中主动提问长城（哈弗、坦克、魏牌、欧拉、长城炮）相关车型（见右侧），收到客服回应后没有表示拒绝，对话完整
• 高意向-同意门店联系：在开场白、车型/优惠介绍阶段，客户有肯定表达，且询问报价/车型配置/优惠活动等信息，配合提供城市/同意门店再联系
• 高意向-主动要求微信/电话再联系-有信息：客户主动提及长城车型，或者AI介绍完长城车型/优惠后，客户再要求加微信/电话联系
• 高意向-交互后同意：在开场白、车型/优惠介绍阶段，客户拒绝不超过1轮（不需要、不考虑、不用了等），最终转向配合提供城市/同意门店再联系
• 高意向-已预约试乘试驾：“已经试驾过了”“我这会就在你们店里”
• 高意向-已加微信/已联系：“我有销售的微信”
• 中意向-敷衍回复：在开场白、车型/优惠介绍阶段，客户仅简短表达肯定（如：嗯、好、行），未主动提问，同意门店再联系，未配合提供城市
• 中意向-有需求-拒绝跟进：客户在对话中主动提问长城（哈弗、坦克、魏牌、欧拉、长城炮）相关车型（见右侧），收到客服回应后拒绝进一步跟进
• 低意向-延迟购车需求：在开场白、车型/优惠介绍阶段，客户未主动提问、未表达肯定，表示过段时间再看
• 低意向-非自主决定：与家人商量；亲友看车
• 低意向-自主决策型：客户表示自己联系
• 低意向-主动要求微信/电话再联系-无信息：开场白打招呼后，客户直接要求加微信/电话联系，没有多轮对话
""".strip()

PROMPT_TEMPLATE = """
请根据客服与客户电话通话记录，首先判断两个说话人哪个是客服工作人员，哪个是客户，然后根据客户在对话过程中表达的含义，按照下面的标签体系进行分类，输出分类结果和对应的标签。
另外请尽量抽取以下三个字段：
1. 意向城市：客户明确提到的买车城市，例如“内蒙赤峰”“福州”“合肥”
2. 购车周期：客户计划多久内买车，例如“过几个月”“明年”“下周”
3. 意向车型：客户提到的品牌/车系/车型，例如“哈弗H6”“哈弗猛龙”“欧拉好猫”“坦克300”

如果这三个字段在对话中没有被明确提到，请输出空字符串，不要猜测。
标签及定义：

{label_set}

通话记录：

{dialogue}

仅输出 JSON 字符串，不要输出任何其他内容，JSON 结构要求如下：

{{
  "audio_file": "",
  "label": "",
  "confidence": 0.0,
  "reason": "",
  "transcript": "",
  "intent_city": "",
  "purchase_timeline": "",
  "intent_model": ""
}}
""".strip()

_TEXT_KEYS = ("text", "sentence", "content", "transcript", "utterance")
_SPEAKER_KEYS = ("speaker", "speaker_id", "speakerId", "spk", "spk_id", "role", "channel")
_TIME_KEYS = ("start_time", "start", "begin_time", "offset", "ts")
_LIST_KEYS = ("utterances", "segments", "sentences", "results", "dialog", "conversation")


class LlmClient:
    """LLM classification with transcript formatting and structured JSON validation."""

    def __init__(
        self,
        settings: LlmSettings,
        api_key: str | None,
        labels_path: Path,
        async_client: AsyncOpenAI | None = None,
    ) -> None:
        self.settings = settings
        self.api_key = api_key
        self.labels_path = labels_path
        self._async_client = async_client
        self._labels_cache: str | None = None

    async def classify(self, oss_path: str, raw_asr_result: dict) -> ClassificationResult:
        if not self.api_key:
            raise RuntimeError("Missing LLM credentials: set LLM_API_KEY in .env")

        transcript = self.format_transcript(raw_asr_result)
        label_set = self._load_label_set()
        prompt = PROMPT_TEMPLATE.format(label_set=label_set, dialogue=transcript)

        try:
            response = await self._create_completion(prompt)
        except Exception as exc:
            if self._is_retryable_exception(exc):
                raise RetryableExternalError(f"LLM request failed: {exc}") from exc
            raise RuntimeError(f"LLM request failed: {exc}") from exc

        content = response.choices[0].message.content or ""
        parsed = self._parse_json_content(content)
        parsed["oss_path"] = oss_path
        parsed["transcript"] = parsed.get("transcript") or transcript
        parsed["intent_city"] = self._normalize_optional_field(parsed.get("intent_city"))
        parsed["purchase_timeline"] = self._normalize_optional_field(
            parsed.get("purchase_timeline")
        )
        parsed["intent_model"] = self._normalize_optional_field(parsed.get("intent_model"))
        parsed["llm_cost"] = self._format_llm_cost(response)
        parsed.pop("audio_file", None)
        return ClassificationResult.model_validate(parsed)

    async def _create_completion(self, prompt: str) -> Any:
        request_kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": "你是一个严格输出 JSON 的客户意向分类助手。",
                },
                {"role": "user", "content": prompt},
            ],
        }
        if self.settings.use_response_format_json:
            try:
                return await self._get_client().chat.completions.create(
                    **request_kwargs,
                    response_format={"type": "json_object"},
                )
            except Exception as exc:
                if self._is_unsupported_response_format(exc):
                    return await self._get_client().chat.completions.create(**request_kwargs)
                raise
        return await self._get_client().chat.completions.create(**request_kwargs)

    def format_transcript(self, raw_asr_result: dict) -> str:
        utterances = self._extract_utterances(raw_asr_result)
        if not utterances:
            raise RuntimeError("ASR result does not contain recognizable utterances")

        lines: list[str] = []
        speaker_aliases: dict[str, str] = {}
        fallback_next = 1
        previous_speaker: str | None = None

        for utterance in utterances:
            text = self._normalize_text(utterance)
            if not text:
                continue
            speaker_token = self._speaker_token(utterance)
            if speaker_token is None:
                speaker_label = f"speaker {fallback_next}"
                fallback_next = 2 if fallback_next == 1 else 1
            else:
                speaker_label = speaker_aliases.setdefault(
                    speaker_token,
                    f"speaker {len(speaker_aliases) + 1}",
                )

            if lines and previous_speaker == speaker_label:
                lines[-1] = f"{lines[-1]} {text}"
            else:
                lines.append(f"{speaker_label}: {text}")
            previous_speaker = speaker_label

        if not lines:
            raise RuntimeError("ASR result utterances are empty after normalization")
        return "\n".join(lines)

    def _load_label_set(self) -> str:
        if self._labels_cache is not None:
            return self._labels_cache
        if self.labels_path.exists():
            content = self.labels_path.read_text(encoding="utf-8").strip()
            if content and "TODO:" not in content:
                self._labels_cache = content
                return content
        self._labels_cache = DEFAULT_LABEL_SET
        return self._labels_cache

    def _get_client(self) -> AsyncOpenAI:
        if self._async_client is None:
            self._async_client = AsyncOpenAI(
                api_key=self.api_key,
                base_url=self.settings.base_url,
                timeout=self.settings.request_timeout_seconds,
            )
        return self._async_client

    def _extract_utterances(self, payload: Any) -> list[dict[str, Any]]:
        direct = self._direct_utterances(payload)
        if direct:
            return sorted(direct, key=self._sort_key)
        if isinstance(payload, dict):
            for key in _LIST_KEYS:
                nested = payload.get(key)
                direct = self._direct_utterances(nested)
                if direct:
                    return sorted(direct, key=self._sort_key)
            for value in payload.values():
                nested = self._extract_utterances(value)
                if nested:
                    return sorted(nested, key=self._sort_key)
        if isinstance(payload, list):
            for item in payload:
                nested = self._extract_utterances(item)
                if nested:
                    return sorted(nested, key=self._sort_key)
        return []

    def _direct_utterances(self, payload: Any) -> list[dict[str, Any]]:
        if not isinstance(payload, list):
            return []
        utterances: list[dict[str, Any]] = []
        for item in payload:
            if not isinstance(item, dict):
                continue
            if any(key in item for key in _TEXT_KEYS):
                utterances.append(item)
        return utterances

    @staticmethod
    def _normalize_text(utterance: dict[str, Any]) -> str:
        for key in _TEXT_KEYS:
            value = utterance.get(key)
            if isinstance(value, str):
                text = " ".join(value.split())
                if text:
                    return text
        return ""

    @staticmethod
    def _speaker_token(utterance: dict[str, Any]) -> str | None:
        for key in _SPEAKER_KEYS:
            value = utterance.get(key)
            if value is None:
                continue
            if isinstance(value, str):
                normalized = value.strip()
                if normalized:
                    return normalized
            if isinstance(value, int):
                return str(value)
        return None

    @staticmethod
    def _sort_key(utterance: dict[str, Any]) -> tuple[float, str]:
        for key in _TIME_KEYS:
            value = utterance.get(key)
            if isinstance(value, (int, float)):
                return float(value), json.dumps(utterance, sort_keys=True, ensure_ascii=False)
            if isinstance(value, str):
                try:
                    return float(value), json.dumps(utterance, sort_keys=True, ensure_ascii=False)
                except ValueError:
                    continue
        return float("inf"), json.dumps(utterance, sort_keys=True, ensure_ascii=False)

    @staticmethod
    def _parse_json_content(content: str) -> dict[str, Any]:
        text = content.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if len(lines) >= 3:
                text = "\n".join(lines[1:-1]).strip()
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"LLM output is not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError("LLM output must be a JSON object")
        return parsed

    @staticmethod
    def _normalize_optional_field(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        return str(value).strip()

    def _format_llm_cost(self, response: Any) -> str:
        input_tokens, output_tokens = self._extract_token_usage(response)
        if input_tokens is None or output_tokens is None:
            return ""
        cost, currency = self.get_doubao_seed_2_0_pro_price(input_tokens, output_tokens)
        return f"{cost:.4f} {currency}"

    @staticmethod
    def _extract_token_usage(response: Any) -> tuple[int | None, int | None]:
        usage = getattr(response, "usage", None)
        if usage is None:
            return None, None

        input_tokens = getattr(usage, "input_tokens", None)
        output_tokens = getattr(usage, "output_tokens", None)

        if input_tokens is None:
            input_tokens = getattr(usage, "prompt_tokens", None)
        if output_tokens is None:
            output_tokens = getattr(usage, "completion_tokens", None)

        if input_tokens is None or output_tokens is None:
            return None, None
        return int(input_tokens), int(output_tokens)

    @staticmethod
    def get_doubao_seed_2_0_pro_price(input_tokens: int, output_tokens: int) -> tuple[float, str]:
        if input_tokens <= 32000:
            input_price = 0.0032 / 1000
        elif input_tokens <= 128000:
            input_price = 0.0048 / 1000
        else:
            input_price = 0.0096 / 1000

        if output_tokens <= 32000:
            output_price = 0.0160 / 1000
        elif output_tokens <= 128000:
            output_price = 0.0240 / 1000
        else:
            output_price = 0.0480 / 1000

        return input_price * input_tokens + output_price * output_tokens, "元"

    @staticmethod
    def _is_retryable_exception(exc: Exception) -> bool:
        status_code = getattr(exc, "status_code", None)
        if status_code in {408, 409, 429}:
            return True
        if isinstance(status_code, int) and status_code >= 500:
            return True
        return exc.__class__.__name__ in {
            "APIConnectionError",
            "APITimeoutError",
            "InternalServerError",
            "RateLimitError",
        }

    @staticmethod
    def _is_unsupported_response_format(exc: Exception) -> bool:
        message = str(exc)
        return (
            "response_format.type" in message and "not supported" in message.lower()
        ) or "json_object" in message and "not supported" in message.lower()
