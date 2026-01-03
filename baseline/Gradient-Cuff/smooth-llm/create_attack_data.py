import json

# 读取第一个JSON文件
with open('/data/SafeDecoding-Attackers/orbench.json', 'r', encoding='utf-8') as f:
    gcg_data = json.load(f)

# 过滤出target-model为vicuna的数据
vicuna_data = [item for item in gcg_data if item.get('source') == 'DeepInception']

# 提取goal和target
goals = []
targets = []

for item in vicuna_data:
    goals.append(item['prompt'])
    targets.append(item['target'])

# 构建新的JSON结构
new_json = {
    "goal": goals,
    "target": targets
}

# 输出结果
print(json.dumps(new_json, indent=2, ensure_ascii=False))


with open('/data/smooth-llm/data/GCG/ORbench.json', 'w', encoding='utf-8') as f:
     json.dump(new_json, f, indent=2, ensure_ascii=False)