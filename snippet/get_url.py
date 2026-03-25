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
