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
