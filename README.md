# call-audio-record-cls

Minimal async scaffold for:

1. iterating call recordings from OSS
2. submitting audio to ASR
3. classifying intent with an LLM
4. writing structured JSONL and CSV outputs

The current version intentionally leaves all OSS, ASR, and LLM SDK calls as TODO placeholders.

## Quick start

```bash
uv sync
uv run call-audio-pipeline --help
uv run call-audio-asr-sample preview --limit 3
uv run call-audio-asr-sample run --limit 3
```

## Environment

Create a `.env` file in the project root for credentials and runtime configuration.

## Small Batch Helpers

Use the small-batch helper before full runs:

- `uv run call-audio-asr-sample preview --limit 3` lists candidate files and whether ASR output already exists.
- `uv run call-audio-asr-sample run --limit 3` runs OSS + ASR only, saves raw ASR JSON to OSS, and writes local artifacts under `output/sample_batch/`.
