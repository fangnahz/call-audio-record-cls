---
name: ASR + Customer Intent Classification Pipeline

description: This skill implements a three-stage pipeline: Stage 1. Perform ASR on customer service call recordings stored in OSS; Stage 2. Use LLM to classify customer intent based on ASR transcripts. The pipeline must be batch-friendly, fault-tolerant, and scalable; Stage 3. Read the LLM results from OSS, and format the final CSV output file for downstream consumption.
---

# ASR + Customer Intent Classification Pipeline

## Workflow

### Stage 1 — ASR Processing

Input OSS Location

oss://yiwise-asr/zhongqi-changcheng-recordings/

Output OSS Location (ASR Results)

oss://yiwise-asr/zhongqi-changcheng-asr/

#### File Iteration Rules

When iterating OSS objects:

• All stored recordings are .mp3 files
• Skip files that already have ASR output in the output bucket
• Use the same filename for input and output, but use .json for the output format
• Processing must be idempotent

Example:

audio-1_48k_1ch.mp3
→ skip if `oss://yiwise-asr/zhongqi-changcheng-asr/audio-1_48k_1ch.json` already exists in ASR bucket

#### Temporary URL Generation

For each audio file:
• Generate a temporary signed URL
• URL must remain valid for the duration of ASR processing
• Do not download audio locally unless required by ASR SDK

```python
from dotenv import load_dotenv
import alibabacloud_oss_v2 as oss

load_dotenv()

credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
cfg = oss.config.load_default()
cfg.credentials_provider = credentials_provider
cfg.region = "cn-hangzhou"
cfg.endpoint = "oss-cn-hangzhou.aliyuncs.com"

client = oss.Client(cfg)

paginator = client.list_objects_v2_paginator()

for page in paginator.iter_page(oss.ListObjectsV2Request(
    bucket="yiwise-asr",
    prefix="zhongqi-changcheng-recordings/"
)):
    for obj in page.contents:
        pre_result = client.presign(oss.GetObjectRequest(bucket="yiwise-asr", key=obj.key))
        file_url = pre_result.url
        print(
            f'method: {pre_result.method},'
            f' expiration: {pre_result.expiration.strftime("%Y-%m-%dT%H:%M:%S.000Z")},'
            f' url: {file_url}'
        )
        break
    break

```

#### Audio Metadata Extraction

Audio format information must be extracted from filename (`obj.key`).

Examples:
• .../audio-1_48k_1ch.mp3 → rate = 48000, channels = 1
• .../audio-56_16k_2ch.mp3 → rate = 16000, channels = 2

Rules:
• _48k_ → 48000 Hz
• _16k_ → 16000 Hz
• \_1ch → mono
• \_2ch → stereo

This metadata must be passed correctly to ASR.

#### Stereo vs Mono Handling

Different recordings have different channel semantics:

Case 1 — Stereo with role separation
• Channel 1: caller
• Channel 2: customer
• Must preserve role separation

Case 2 — Stereo duplicated mixed audio
• Both channels identical
• Treat as mono

Case 3 — Mono mixed audio
• Single track
• ASR must infer speakers if possible

Implementation must detect and handle these cases properly.

#### ASR Invocation

For each audio:
• Call ASR using signed URL
• Pass correct:
• sample rate
• channel count
• language (if required)

```python
import json
import os
import re
import time
import uuid

import requests
from dotenv import load_dotenv

load_dotenv()


def submit_task():

    submit_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/submit"

    task_id = str(uuid.uuid4())

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Api-Sequence": "-1",
    }

    rate_channel_ptn = r"audio-\d+_(\d+)k_([1-2])ch\.mp3"
    rate, channel = re.search(rate_channel_ptn, file_url).groups()
    rate = int(rate) * 1000
    channel = int(channel)

    request = {
        "user": {"uid": "fake_uid"},
        "audio": {
            "url": file_url,
            "format": "mp3",
            # "codec": "map3",
            "rate": rate,
            "channel": channel
        },
        "request": {
            "model_name": "bigmodel",
            "enable_channel_split": True,
            "enable_ddc": True,
            "enable_speaker_info": True,
            "enable_punc": True,
            "enable_itn": True,
            "corpus": {
                "correct_table_name": "",
                "context": "",
            },
        },
    }
    print(f"Submit task id: {task_id}")
    response = requests.post(submit_url, data=json.dumps(request), headers=headers)
    if (
        "X-Api-Status-Code" in response.headers
        and response.headers["X-Api-Status-Code"] == "20000000"
    ):
        print(
            f"Submit task response header X-Api-Status-Code: {response.headers['X-Api-Status-Code']}"
        )
        print(
            f"Submit task response header X-Api-Message: {response.headers['X-Api-Message']}"
        )
        x_tt_logid = response.headers.get("X-Tt-Logid", "")
        print(
            f"Submit task response header X-Tt-Logid: {response.headers['X-Tt-Logid']}\n"
        )
        return task_id, x_tt_logid
    else:
        print(f"Submit task failed and the response headers are: {response.headers}")
        exit(1)
    return task_id


def query_task(task_id, x_tt_logid):
    query_url = "https://openspeech-direct.zijieapi.com/api/v3/auc/bigmodel/query"

    headers = {
        "X-Api-App-Key": appid,
        "X-Api-Access-Key": token,
        "X-Api-Resource-Id": "volc.bigasr.auc",
        "X-Api-Request-Id": task_id,
        "X-Tt-Logid": x_tt_logid,  # 固定传递 x-tt-logid
    }

    response = requests.post(query_url, json.dumps({}), headers=headers)

    if "X-Api-Status-Code" in response.headers:
        print(
            f"Query task response header X-Api-Status-Code: {response.headers['X-Api-Status-Code']}"
        )
        print(
            f"Query task response header X-Api-Message: {response.headers['X-Api-Message']}"
        )
        print(
            f"Query task response header X-Tt-Logid: {response.headers['X-Tt-Logid']}\n"
        )
    else:
        print(f"Query task failed and the response headers are: {response.headers}")
        exit(1)
    return response


def main():
    task_id, x_tt_logid = submit_task()
    while True:
        query_response = query_task(task_id, x_tt_logid)
        code = query_response.headers.get("X-Api-Status-Code", "")
        if code == "20000000":  # task finished
            print(query_response.json())
            print("SUCCESS!")
            exit(0)
        elif code != "20000001" and code != "20000002":  # task failed
            print("FAILED!")
            exit(1)
        time.sleep(1)


file_url = "https://yiwise-asr.oss-cn-hangzhou.aliyuncs.com/zhongqi-changcheng-recordings/audio-100_48k_1ch.mp3?x-oss-signature-version=OSS4-HMAC-SHA256&x-oss-date=20260324T070848Z&x-oss-expires=900&x-oss-credential=LTAICuX7MOmHW6Oy%2F20260324%2Fcn-hangzhou%2Foss%2Faliyun_v4_request&x-oss-signature=29e9b6e550e2e3eb29082858d1fe25822000a7cc062bc5d580b48032b27c211f"

# 填入控制台获取的app id和access token
appid = os.getenv("ASR_APP_ID")
token = os.getenv("ASR_TOKEN")

if __name__ == "__main__":
    main()

```

#### Save ASR Result

ASR output must be:

• Raw json result
• No post-processing
• No role relabeling
• No formatting

The raw ASR result must be saved as-is.

• Save one OSS object per audio
• Use same filename as input audio, but with .json extension
• Content = raw ASR response
• No additional metadata

```python
import json

from dotenv import load_dotenv
import alibabacloud_oss_v2 as oss

load_dotenv()

credentials_provider = oss.credentials.EnvironmentVariableCredentialsProvider()
cfg = oss.config.load_default()
cfg.credentials_provider = credentials_provider
cfg.region = "cn-hangzhou"
cfg.endpoint = "oss-cn-hangzhou.aliyuncs.com"

client = oss.Client(cfg)

paginator = client.list_objects_v2_paginator()

for page in paginator.iter_page(oss.ListObjectsV2Request(
    bucket="yiwise-asr",
    prefix="zhongqi-changcheng-recordings/"
)):
    for obj in page.contents:
        pre_result = client.presign(oss.GetObjectRequest(bucket="yiwise-asr", key=obj.key))
        file_url = pre_result.url
        # get asr result as demonstrated earlier, here we'll just use a dummy string to show how to upload item to oss
        # query_response = query_task(task_id, x_tt_logid)
        # asr_result = query_response.json()
        key = obj.key.replace("zhongqi-changcheng-recordings/", "zhongqi-changcheng-asr/").replace(".mp3", ".json")
        asr_result = {"result": "this is a dummy asr result 仅用于示例"}
        body = json.dumps(asr_result, ensure_ascii=False).encode("utf-8")
        result = client.put_object(oss.PutObjectRequest(
            bucket="yiwise-asr",
            key=key,
            body=body
        ))
        print(
            f'status code: {result.status_code},'
            f' request id: {result.request_id},'
            f' content md5: {result.content_md5},'
            f' etag: {result.etag},'
            f' hash crc64: {result.hash_crc64},'
            f' version id: {result.version_id},'
        )
        break
    break

```

### Stage 2 — LLM Classification

#### Read ASR Results

• Iterate ASR result bucket
• Load ASR content
• Skip malformed or empty results
• Continue processing even if some fail

#### Transcript Formatting

Convert ASR output into conversation format:

Example:

speaker 1: ...
speaker 2: ...
speaker 1: ...

Rules:
• Preserve chronological order
• Preserve role separation if available
• If roles unavailable, assign speaker 1 / speaker 2 heuristically

#### Classification Task

The LLM must classify customer intent.

Following are the output label set and the explanation for each label:

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

Label definitions must be provided in the prompt.

#### LLM Invocation

• Use structured JSON output
• Do not allow free-form output
• Include:
  • transcript
  • label definitions
  • classification instructions

```python
import os

from dotenv import load_dotenv
from openai import OpenAI
from volcenginesdkarkruntime import Ark

load_dotenv()

api_key = os.getenv('ARK_API_KEY')

client = Ark(
    base_url='https://ark.cn-beijing.volces.com/api/v3',
    api_key=api_key,
)

TEMPLATE = """
请根据客服与客户电话通话记录，首先判断两个说话人哪个是客服工作人员，哪个是客户，然后根据客户在对话过程中表达的含义，按照下面的标签体系进行分类，输出分类结果和对应的标签：
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
    "transcript": ""
}}

"""

LABEL_SET = """
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
"""

DIALOGUE = """
SPEAKER_1: 你好，这里是长城汽车，请问有什么可以帮您？
SPEAKER_2: 你好，我想问一下你们现在有什么优惠活动吗
SPEAKER_1: 现在我们有一些优惠活动，您是想了解一下我们哪款车型的优惠吗？
SPEAKER_2: 我想问一下哈弗H6现在有什么优惠吗
SPEAKER_1: 哈弗H6现在有一些优惠活动，您可以享受购车补贴和金融优惠。购车补贴方面，您可以获得最高5000元的购车补贴。金融优惠方面，我们提供低至0利率的贷款方案，最长可达5年。您还有其他关于哈弗H6的问题吗？
SPEAKER_2: 这个优惠活动持续到什么时候啊
SPEAKER_1: 这个优惠活动持续到今年年底，您可以在此期间享受这些优惠。您还有其他关于哈弗H6的问题吗？
SPEAKER_2: 没有了，谢谢
"""

system_content = TEMPLATE.format(
    label_set=LABEL_SET,
    dialogue=DIALOGUE
)

response = client.responses.create(
    model="ep-20260217133005-kx6d7",
    input=[
        {
            "role": "system",
            "content": [
                {
                    "type": "input_text",
                    "text": system_content
                }
            ],
        }
    ]
)

print(response.model_dump_json())

```

#### Output Schema

Each classification result must follow:

```json
{
    "audio_file": "",
    "label": "",
    "confidence": 0.0,
    "reason": "",
    "transcript": ""
}
```

#### Save Classification Results

Classification output must be saved to OSS:

Output bucket:
oss://yiwise-asr/zhongqi-changcheng-llm/

Rules:
• One object per audio
• Same filename as original
• JSON content

### Stage 3 — Output Formatting

Read LLM classification results from OSS, and format the final CSV output file for downstream consumption.

## Other notes

### Concurrency Strategy

Processing must support batching.

Recommended limits:
• ASR concurrency: 10
• LLM concurrency: 5

Use:
• asyncio
• semaphore
• retry with backoff

### Retry Rules

Retry:
• network errors
• 5xx responses
• timeout
• rate limit

Do NOT retry:
• invalid input
• malformed audio

Max retries: 3

### Fault Tolerance

• One file failure must not stop pipeline
• Log failures
• Continue processing
• Report summary at end

### Idempotency

The pipeline must:
• Skip already processed ASR files
• Be safe to rerun

### Performance Requirements

• Must handle 100+ recordings
• Must scale to thousands
• Must not load all audio into memory
• Must process incrementally

### Expected End-to-End Flow

OSS audio
↓
Generate signed URL
↓
ASR
↓
Save raw ASR
↓
Format transcript
↓
LLM classification
↓
Save structured result in output CSV file
