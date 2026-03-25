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
