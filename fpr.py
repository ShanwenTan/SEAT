import json

# 读取JSON文件
with open('/data/saffron/outputs/ORbench_Vicuna_0.75_0.8_memory_0.95/ORbench~trade024tp0.8/data/saffron/models/Saffron-1-1B~PrefillAtk10~Vicuna_7B_Inst~LlamaGuard3_1B~LlamaGuard3_1B/safety_evaluation_results.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 统计包含特定文本的条目数量
count = 0
for result in data['results']:
    if "Due to security policy, I cannot answer this question" in result['output_text']:
        count += 1

# 输出结果
print(f"包含'Due to security policy, I cannot answer this question'的条目数量: {count}")