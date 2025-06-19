# coding: utf-8
import os
import sys
import argparse
import platform
import numpy as np
import torch
import pandas as pd
import re
import json


import transformers

from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoTokenizer, AutoModelForCausalLM


generation_config = dict(
    temperature=0.7,
    top_k=40,
    top_p=0.6,
    do_sample=False,
    num_beams=1,
    repetition_penalty=1.0,
    # no_repeat_ngram_size=4,
    # encoder_no_repeat_ngram_size=4,
    max_new_tokens=1024
)


model_path = 'DeepSeek-R1-Distill-Qwen-7B-GRPO'



tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

model = AutoModelForCausalLM.from_pretrained(
    model_path, 
    torch_dtype=torch.bfloat16,
    device_map='auto',
    trust_remote_code=True,
)

model.eval()

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


#######################################################################################
# 使用pyarrow引擎
df = pd.read_parquet('data/gsm8k/main/test-00000-of-00001.parquet', engine='pyarrow')
question_list = df["question"]
answer_list = df["answer"]
print(len(question_list))

total = 0
fw = open("temp/result.jsonl", 'w', encoding='utf-8')
for question, answer in zip(question_list, answer_list):
    messages = [
        {'role': 'system', 'content': SYSTEM_PROMPT},
        {'role': 'user', 'content': question},  # q1
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = tokenizer([text], return_tensors="pt").to('cuda')
    length = model_inputs.input_ids.shape[1]
    generation_output = model.generate(
        input_ids = model_inputs.input_ids,
        **generation_config
    )


    output_ids = generation_output.cpu().numpy()[0][length:].tolist()
    output = tokenizer.decode(output_ids, skip_special_tokens=True)

    gold = answer.split('####')[1].replace(',', '').replace('$', '').strip()
    match = re.search(r'<answer>(.*?)</answer>', output)
    true_total = 0
    flag = False
    if match:
        answer_content = match.group(1).strip()  # 提取并去除前后空格
        if answer_content == gold:
            true_total = true_total + 1
            flag = True
        else:
            pass
    print("="*20)
    print("num:", total)
    print(output)
    if flag:
        print("%"*5)
        print(gold)
    total = total + 1

    data = {"question": question, "answer": answer, "output": output, "good":flag}
    fw.write(json.dumps(data, ensure_ascii = False) + '\n')
    fw.flush()

print("*"*100)
print(true_total/len(question_list))