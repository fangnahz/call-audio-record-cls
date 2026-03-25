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
LLM_CONCURRENCY = 20
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
