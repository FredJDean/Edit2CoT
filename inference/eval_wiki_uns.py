from tqdm import tqdm
from edit_cot import get_result
from transformers import AutoTokenizer
from transformers import StoppingCriteria, AutoModel
from vllm import LLM
import json
import argparse
import multiprocessing

instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."

def jaccard_sim(str1, str2):
    set1 = set(str1)
    set2 = set(str2)
    return len(set1 & set2) / len(set1 | set2)

def eval_efficacy(model_edit, llmtokenizer, dataset):
    result = []
    tot = 0
    cor = 0

    for d in tqdm(dataset):
        tot += 1
        requested_rewrite = d["requested_rewrite"]
        target_new = requested_rewrite['answer_new']
        edit_fact = requested_rewrite['fact_new_uns']
        question = requested_rewrite['question']

        ques = "Edit Context:" + edit_fact + "\n" + "Question:" + question
        res = get_result(instruct, ques, model_edit, llmtokenizer)
        result.append(res)
        ans = res["answer"]

        print(ans, "ground:", target_new)

        sim = jaccard_sim(ans, target_new)

        # 如果新的回答与新编辑的事实一致
        # 由于wiki这个数据集上下文描述中有很多关于事实描述的变体（甚至还有跨语言的）
        # 因此不能一概而论
        if (
            (ans == target_new)
            or (target_new in ans)
            or (ans in target_new)
            or (sim > 0.7)
        ):
            cor += 1
        print(f'acc = {cor / tot:.4f} ({cor} / {tot})')
    acc = cor / tot
    return acc

def main(args):
    model_name = args.model_name
    editor_path = args.editor_path

    # 加载分词器和模型
    llmtokenizer = AutoTokenizer.from_pretrained(model_name)
    model_edit = LLM(
        model=editor_path,
        tokenizer=model_name,
        tensor_parallel_size=args.tensor_parallel_size,  # 根据GPU设置调整
        dtype="float16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )

    # 加载数据集
    dataset = json.load(open(args.data_path, "r"))

    # 评估 efficacy
    efficacy_score = eval_efficacy(model_edit, llmtokenizer, dataset)
    efficacy = {
        "model": model_name,
        "efficacy": efficacy_score
    }

    with open(args.output_filename, "a", encoding="utf-8") as f:
        json.dump(efficacy, f, ensure_ascii=False, indent=2)
        f.write("\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="/gemini/space/fujinhu/pretrain-models/llama-3-8b-instruct")
    parser.add_argument("--data_path", type=str, default="../data/wikiUpdate.json")
    parser.add_argument("--editor_path", type=str, default="/gemini/space/fujinhu/MyEdit/train_grpo/log/grpo_llama")
    parser.add_argument("--output_filename", type=str, default="./output/output_wiki_uns.json")
    parser.add_argument("--max_iter", type=int, default=4)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    args = parser.parse_args()

    multiprocessing.set_start_method('spawn', force=True)

    main(args)

