import json


with open('/ORbench_Vicuna_outputs.json', 'r', encoding='utf-8') as file:
    data = json.load(file)


count = 0
for result in data['results']:
    if "Due to security policy, I cannot answer this question" in result['output_text']:
        count += 1



print(f"'Due to security policy, I cannot answer this question': {count}")
