import json
import openai
from openai import OpenAI
import time
from tqdm import tqdm
import re

# 生成数据的提示
prompt_construct = ''' Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**, a few examples of which are provided below:\n
Edit Fact: The author of Murder in the Cathedral is Terrance Dicks. \n
Question: Who is the author of Murder in the Cathedral?\n
<think>Step1: I need to know the author of Murder in the Cathedral. Knowledge1: According to the Edit Fact, the author of Murder in the Cathedral is Terrance Dicks.</think>\n
<answer>Terrance Dicks</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: The president of the United States of America is LeBron James.\n
Question: Who is the president of the United States of America?\n
<think>Step1: I need to know the president of the United States of America. Knowledge1: According to the Edit Fact, the president of the United States of America is LebBron James.</think>\n
<answer>Lebron James</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: Carl Sagan is employed by British Broadcasting Corporation. \n 
Question: What is the employer of the spouse of Ann Druyan? \n
<think>Step1: I need to know the spouse of Ann Druyan. Knowledge1: According to my knowledge, the spouse of Ann Druyan is Carl Sagan. Step2: I need to know what is the employer of Carl Sagan. Knowledge2: According to Edit Fact, Carl Sagan is employed by British Broadcasting Corporation.</think>\n
<answer>British Broadcasting Corporation</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: The name of the current head of state in United States of America is Connachta.\n
Question: Who is the current head of state of the country where Josiah Whitney holds citizenship?\n
<think>Step1: I need to know where Josiah Whitney holds citizenship? Knowledge1:According to my knowledge, Josiah Whitney holds citizenship of United States of America. Step2: I need to know who is the current head of state in United States of America. Knowledge2: According to the Edit Fact, the name of the current head of state in United States of America is Connachta.</think>\n
<answer>Connachta</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: The author of The French Lieutenant's Woman is William Gibson.
Question: Which country is the author of \"The French Lieutenant's Woman\" a citizen of?
<think>Step1: I need to know the author of \"The French Lieutenant's Woman\". Knowledge1: According to the Edit Fact, the author of The French Lieutenant's Woman is William Gibson. Step2: I need to know the country of William Gibson. Knowledge2: According to my knowledge, the the country of William Gibson is United States of America.</think>\n
<answer>United States of America</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: The official language of Spain is Arabic.
Question: What is the official language of the country where Shaun Livingston's sport was originated?\n
<think>Step1: I need to know the sport of Shaun Livingston. Knowledge1: According to my knowledge, Shaun Livingston is a basketball player. Step2: I need to know which country was the sport of basketball originated? Knowledge2: According to my knowledge, the sport of basketball was originated in Spain. Step3: I need to know the official language used in Spain. Knowledge3: According to the Edit Fact, the official language of Spain is Arabic.</think>\n
<answer>Arabic</answer>\n\n

Your task is to break down the question into steps and extract the chain of thought based on the **Editing Facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Fact**, a few examples of which are provided below:\n
Edit Fact: {ef}\n
Question: {ques}\n
'''

with open("data/MQuAKE-CF.json") as f:
    data = json.load(f)

# 对数据进行处理
new_json = []
for item in data:
    requested_rewrite = item["requested_rewrite"][0]
    subject = requested_rewrite["subject"]
    target_new = edit_fact = requested_rewrite['target_new']['str']
    edit_fact = requested_rewrite["prompt"].format(subject) + ' ' + target_new
    questions = item['questions']
    answer = item['new_answer']
    answers_list = item['new_answer_alias']
    for ques in questions:
        new_json.append({'question': ques, 'edit_fact': edit_fact, "answer": answer, "answers_list": answers_list})

client = OpenAI(api_key='sk-4ecdf657ff28487382351bfb952d70ca',
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")


# 获得chatgpt的回复
def getResponse(prompt, max_retries=10):
    retries = 0
    while retries < max_retries:
        try:
            res = client.chat.completions.create(
                model='qwen-max',
                messages=[
                    {'role': 'user', 'content': prompt}
                ],
                temperature=0,
            )
            return res.choices[0].message.content
        except Exception as e:
            print("An error occured:", e)
            print('Retrying...')
            retries += 1
            time.sleep(5)
    return ''


instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."
# 遍历数据集
dataset = []
for sample in tqdm(new_json):
    question = sample['question']
    edit_fact = sample['edit_fact']
    answer_list = sample['answers_list']
    answer_new = sample['answer']
    knowledge = getResponse(prompt_construct.format(ques=question, ef=edit_fact), max_retries=10)
    match = re.search(r'<answer>(.*?)</answer>', knowledge)
    result = match.group(1)
    if result in answer_list or result == answer_new:
        dataset.append({"Instruct": instruct, "Input": "Question: {}\n Edit Fact: {}".format(question, edit_fact),
                        "Output": knowledge})

# 将数据写入json文件
json_str = json.dumps(dataset, ensure_ascii=False, indent=4)
with open('data/train_data.json', 'w', encoding='utf-8') as f:
    f.write(json_str)