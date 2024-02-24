from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("./chatglm2-6b", trust_remote_code=True)
model = AutoModel.from_pretrained("./chatglm2-6b", trust_remote_code=True).half().cuda()
model = model.eval()
import pandas as pd
import json
import os

df = pd.read_csv('selected-ann.csv')
rule = "请修订题为“我最喜欢的人”的作文。\n修订前作文内容："
for time in [1, 2, 3, 4, 5]:
    for index, row in df.iterrows():
        print(time, index)
        # if file exists, skip
        if os.path.exists('result/enhanced' + '/{}-{}.json'.format(index, time)):
            print("skipped")
            continue
        else:            
            response, history = model.chat(tokenizer, rule + row['content'] + "\n修订后作文内容：", history=[])
            json.dump({'response': response, 'history': history}, open('result/enhanced' + '/{}-{}.json'.format(index, time), 'w', encoding='utf8'), ensure_ascii=False)
            print("generated")