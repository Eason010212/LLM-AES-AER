from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()
import pandas as pd
import json
import os

df = pd.read_csv('selected-ann.csv')
rule = '''请对命题作文评分，作文题为“我最喜欢的人”。评分结果为0-5分，具体评分标准如下。
5分：没有词汇和语法错误，用词丰富多样，能够灵活运用多种句式；表达流畅、连贯，过渡自然，内容组织有条理；表达完全符合主题，能够有效传达信息。
4分：几乎没有词汇和语法，用词较为多样，能够灵活运用多种句式；表达比较流畅自然，内容组织比较有条理；表达比较符合主题，能够传达一定信息。
3分：词汇和语法错误较少，用词多为简单词汇，能够正确运用不同句式；表达基本流畅，但偶尔过渡有些生硬；表达基本符合主题，基本能够被理解。
2分：词汇和语法错误较多，但不至于影响整体理解，能够正确运用基本句式；表达不太流畅，多处前后不连贯；表达基本符合主题，基本能够被理解。
1分：有大量词汇和语法错误，严重影响理解。或字数较少，仅能体现简单用词和基本句式；表达不流畅，多处前后不连贯；表达基本符合主题，有几处难以理解。
0分：字数严重不足，小于40字；表达不流畅，前后不连贯，难以形成连续语义；表达不符合主题。
你理解了这一评分标准吗？接下来，我将给你学生的作答，请你对它进行评分。
评分格式如下，其中[MASK1]应该是一个在0到5之间的整数，代表你的评分，[MASK2]是你对于评分的解释：
总分：[MASK1]分，理由：[MASK2]'''
rule_back = "当然，我理解你的评分标准。现在，你可以给我学生的作答，我会根据你的标准进行评分。"    
prompt = "请你对这篇作文进行评分（0-5分）。"
prompt1 = "假如你是一位资深的语文教师，请你对这篇作文进行评分（0-5分）。"
prompt2 = "严格按照评分标准，请你对这篇作文进行评分（0-5分）。"
for time in [1, 2, 3, 4, 5]:
    for index, row in df.iterrows():
        promptIndex = 0
        for prompt in [prompt, prompt1, prompt2]:
            print(time, index, promptIndex)
            # if file exists, skip
            if os.path.exists('result/chatglm2-6b-shot0-prompt' + str(promptIndex) + '/{}-{}.json'.format(index, time)):
                promptIndex += 1
                print("skipped")
                continue
            else:            
                response, history = model.chat(tokenizer, prompt + row['content'], history=[[rule, rule_back]])
                json.dump({'prompt': prompt + row['content'], 'response': response, 'history': history}, open('result/chatglm2-6b-shot0-prompt' + str(promptIndex) + '/{}-{}.json'.format(index, time), 'w', encoding='utf8'), ensure_ascii=False)
                promptIndex += 1
                print("generated")