from tqdm import tqdm
from edit_cot import retrieve_facts, get_result, get_sent_embeddings
from transformers import AutoTokenizer
from transformers import StoppingCriteria, AutoModel
from vllm import LLM
import json
import argparse
import multiprocessing

instruct = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>. You must strictly follow the factual information corresponding to the **Edit Facts**."

def eval_efficacy(model_edit, llmtokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    tot = 0
    cor = 0

    for d in tqdm(dataset):
        tot += 1
        question = d['src']
        target_new = d['answers'][0]

        fact_ids, fact_value = retrieve_facts(question, fact_embs, contriever, tokenizer)

        edit_fact = new_facts.get(fact_docs[fact_ids[0]]) + ". "
        ques = "Edit Fact:" + edit_fact + "\n" + "Question:" + question  # 上下文知识注入

        res = get_result(instruct, ques, model_edit, llmtokenizer)
        ans = res["answer"]

        if ans == target_new:
            cor += 1

        print(f'acc={cor / tot}({cor}/{tot})')

    acc = cor / tot
    return acc


def eval_rephrase(model_edit, llmtokenizer, dataset, fact_embs, fact_docs, new_facts, contriever, tokenizer):
    tot = 0
    cor = 0
    for d in tqdm(dataset):
        tot += 1
        target_new = d['answers'][0]
        question = d["rephrase"]

        fact_ids, fact_value = retrieve_facts(question, fact_embs, contriever, tokenizer)
        edit_fact = new_facts.get(fact_docs[fact_ids[0]])
        ques = "Edit Fact:" + edit_fact + "\n" + "Question:" + question  # 上下文知识注入

        res = get_result(instruct, ques, model_edit, llmtokenizer)
        ans = res["answer"]

        if ans == target_new or ans in d['answers']:
            cor += 1

        print(f'acc={cor / tot}({cor}/{tot})')

    acc = cor / tot
    return acc

instruct_loc = "Your task is to break down the question into steps and extract the chain of thought based on the **editing facts** into <think></think> tags, and finally get the corresponding answer and put it in <answer></answer>."

def eval_locaility(model_edit, llmtokenizer, dataset):
    result = []
    tot = 0
    cor = 0

    for d in tqdm(dataset):
        tot += 1
        q = d['src']
        target_new = d['answers'][0]

        edit_fact = q + target_new
        question = d["loc"] + "?"
        ques = "Edit Fact:" + edit_fact + "\n" + "Question:" + question  # 上下文知识注入

        # get_result 应该是处理带上下文注入的 LLM 推理函数
        res = get_result(instruct_loc, ques, model_edit, llmtokenizer)
        ans = res["answer"]

        # 不被 edit 上下文影响
        if ans != target_new:
            cor += 1

        print(f'acc={cor / tot}({cor}/{tot})')

    acc = cor / tot
    return acc


def main(args):
    model_name = args.model_name
    editor_path = args.editor_path

    # 分词器
    llmtokenizer = AutoTokenizer.from_pretrained(model_name)

    model_edit = LLM(
        model=editor_path,
        tokenizer=model_name,
        tensor_parallel_size=4,  # adjust based on your GPU setup
        dtype="float16",
        gpu_memory_utilization=0.9,
        enable_prefix_caching=True,
    )

    # 加载数据集
    dataset = json.load(open(args.data_path, "r"))

    # 加载检索器
    contriever = AutoModel.from_pretrained(args.retriever_path).cuda()
    tokenizer = AutoTokenizer.from_pretrained(args.retriever_path)

    # 构建新事实映射
    new_facts = {}
    for d in dataset:
        question = d['src']
        target_new = d['answers'][0]
        target_true = d['pred']
        old_fact = question + target_true
        new_fact = question + target_new
        new_facts[old_fact] = new_fact

    # 提取所有事实句子并嵌入
    all_facts = set()
    for k in new_facts:
        all_facts.add(k)
    all_facts = list(all_facts)

    embs = get_sent_embeddings(all_facts, contriever, tokenizer)

    # 执行评估
    efficacy = eval_efficacy(model_edit, llmtokenizer, dataset, embs, all_facts, new_facts, contriever, tokenizer)
    paraphrase = eval_rephrase(model_edit, llmtokenizer, dataset, embs, all_facts, new_facts, contriever, tokenizer)
    loc = eval_locaility(model_edit, llmtokenizer, dataset)

    # 保存结果
    result = {
        "model": model_name,
        "efficacy": efficacy,
        "paraphrase success": paraphrase,
        "locality": loc
    }

    with open(args.output_filename, "a", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_name",
        type=str,
        default="/gemini/space/fujinhu/pretrain-models/falcon3-10B-Instruct"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data/zsre_mend_eval.json"
    )
    parser.add_argument(
        "--editor_path",
        type=str,
        default="../train_sft/output/output_falcon_sft"
    )
    # 检索工具
    parser.add_argument(
        "--retriever_path",
        type=str,
        default="/gemini/space/fujinhu/pretrain-models/facebook/contriever-msmarco"
    )
    parser.add_argument(
        "--output_filename",
        type=str,
        default="./output/output_zsre.json"
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=4
    )

    args = parser.parse_args()
    multiprocessing.set_start_method('spawn', force=True)
    main(args)
