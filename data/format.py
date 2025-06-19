import json

input_file = "multi_hop_reason.jsonl"  # 输入文件路径
output_file = "generated_multi_hop_data.jsonl" # 输出文件路径

with open(input_file, 'r', encoding='utf-8') as fin, open(output_file, 'w', encoding='utf-8') as fout:
    for line in fin:
        data = json.loads(line)
        if "Instruct" in data:
            # 替换字段中的 **editing facts** 为 **Edit Fact**
            data["Instruct"] = data["Instruct"].replace("**editing facts**", "**Edit Fact**")
        fout.write(json.dumps(data, ensure_ascii=False) + '\n')
