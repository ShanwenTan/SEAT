import json

# 读取原始JSON文件
with open('/data/SafeDecoding-Attackers/orbench.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 提取目标条目
llama2_goals = []
for item in data:
    if item.get("source") == "DeepInception":
        llama2_goals.append({"user_query": item["prompt"]})

# 保存为新的JSON文件
with open('attack_data/ORbench.json', 'w', encoding='utf-8') as f:
    for goal in llama2_goals:
        json_line = json.dumps(goal, ensure_ascii=False)
        f.write(json_line + '\n')

print(f"成功提取 {len(llama2_goals)} 个llama2目标的查询")
