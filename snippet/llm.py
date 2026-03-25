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

```json
{{
    "audio_file": "",
    "label": "",
    "confidence": 0.0,
    "reason": "",
    "transcript": ""
}}
```

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
