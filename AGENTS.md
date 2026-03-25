# AGENTS.md

## Project Purpose

This project processes customer service call recordings stored in OSS.
The pipeline:

Step 1: ASR

- Iterate audio files from OSS
- Skip files that are already processed
- Generate temporary access URLs
- Perform ASR
- Save raw ASR results back to OSS

Step 2: LLM classification

- Read ASR results from OSS
- Format transcripts into conversation form
- Classify customer intent according to label definitions
- Save the LLM results in OSS

Step 3: Format CSV

- Read LLM results from OSS
- Format and save output CSV file in the project root folder

Step 1, ASR:

1. Iterate audio files from OSS (`oss://yiwise-asr/zhongqi-changcheng-recordings/`)
   1. Skip files that already have the ASR output saved in the OSS (`oss://yiwise-asr/zhongqi-changcheng-recordings-asr/`).
1. Generate temporary access URLs
1. Perform ASR to obtain transcripts
   1. Read the name of the audio for rate and channel. This step is very important because the audio is collected from different sources and the formats are not the same, following are two examples:
      1. "audio-1_48k_1ch.mp3": rate is 48000, channel is 1 (mono)
      2. "audio-56_16k_1ch.mp3": rate is 16000, channel is 2 (stereo)
   1. Some stereo audio files have one channel for the caller, and one channel for the customer, which is the ideal format. But some stereo audio recordings are just the same mixed audio track repeated for both channels. And lots of other audio recordings are mixed track mono audio files. Treat different type properly and make sure the role separation is done properly.
1. Save ASR results to OSS (`oss://yiwise-asr/zhongqi-changcheng-recordings-asr/)
   1. Save each ASR result of an audio as a separate OSS item, use the same name as the mp3 file
   1. Save unprocessed result as content of the item, any post process should be performed in the next step.

Step 2, LLM classification:

1. Read the ASR results from OSS
   1. Process the ASR result, and format the transcript into a conversation format, with proper role labels like "speaker 1" and "speaker 2".
1. Use LLM to classify customer intent (e.g. willingness to pay)
1. Save structured results to OSS

The system must support batch processing and scale to large numbers of recordings.

Step 3, format CSV output file

## Python & Environment

• Python version: 3.13
• Dependency manager: uv
• Virtual environment must be managed by uv

## Secrets Management

• Secrets is stored in .env
• .env must NOT be committed

Use python-dotenv to load them.

## Project Structure

```bash
repo/
├─ AGENTS.md
├─ skills/
├─ src/
│ ├─ oss/
│ ├─ asr/
│ ├─ llm/
│ ├─ pipeline/
│ └─ models/
├─ scripts/
├─ tests/
├─ pyproject.toml
└─ .env
```

## Coding Style

• Use asyncio for concurrency
• Prefer async HTTP clients (httpx / aiohttp)
• Avoid thread-based concurrency unless required by SDK
• Use type hints everywhere
• Prefer dataclasses or Pydantic models for structured data

Example:

```python
@dataclass
class ClassificationResult:
    oss_path: str
    transcript: str
    label: str
    confidence: float
```

## Concurrency Guidelines

• Use asyncio.Semaphore to control concurrency
• Separate limits for:
• OSS requests
• ASR requests
• LLM requests
• Avoid unbounded parallelism
• Default safe limits:

```python
ASR_CONCURRENCY = 10
LLM_CONCURRENCY = 5
```

## Retry Strategy

All external calls must implement retries:
• exponential backoff
• max retries: 3
• retry on:
• network errors
• 5xx responses
• rate limit errors

Do NOT retry on 4xx (except 429).

## Logging

• Use structured logging
• Each audio file should log:
• file id
• transcript success/failure
• classification result
• latency

Avoid print statements in production code.

## Error Handling

• Processing must be fault tolerant
• One failed audio must not stop the batch
• Collect failures and report at end

## Output Format

The pipeline must produce structured JSON output:

{
"oss_path": "",
"transcript": "",
"label": "",
"confidence": 0.0,
"reason": ""
}

Batch outputs should be saved as JSONL

## Performance Goals

• Must handle 100+ recordings efficiently
• Must scale to thousands
• Must support configurable concurrency
• Avoid loading all audio into memory

## Testing Strategy

Before running full batch:
• test on 3–5 audio files
• verify:
• ASR output quality
• classification correctness
• output schema

## Dependencies

Preferred libraries:
• httpx (async HTTP)
• asyncio
• pydantic (optional but recommended)
• python-dotenv
• tenacity (for retry)

Avoid heavy frameworks.

## LLM Usage Rules

• Always request structured JSON output
• Include label definitions in prompts
• Do not rely on free-form text classification
• Validate model output before saving

## General Rules for Agents

• Do not introduce unnecessary frameworks
• Keep code simple and modular
• Prefer readability over abstraction
• Optimize only after correctness
• Ensure idempotent batch processing
• Avoid global mutable state

## Done Criteria

The implementation is complete when:

• OSS iteration works
• ASR integration works
• LLM classification works
• Concurrency is controlled
• Retries implemented
• JSONL output generated
• Tested on sample batch
• Runs end-to-end without manual intervention
