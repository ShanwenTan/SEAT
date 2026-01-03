from transformers import AutoTokenizer

# 加载Llama-3-8B-Instruct模型的tokenizer（保持不变）
llama_8b_model_path = "/data/saffron/local_model_cache/LLM-Research/Meta-Llama-3-8B-Instruct"
llama_8b_tokenizer = AutoTokenizer.from_pretrained(llama_8b_model_path)

# 加载Llama-3.2-1B-Instruct模型的tokenizer（更新为指定路径）
llama_1b_model_path = "/data/saffron/local_model_cache/AI-ModelScope/vicuna-7b-v1.5"
llama_1b_tokenizer = AutoTokenizer.from_pretrained(llama_1b_model_path)

# 给定的Llama-3-8B-Instruct模型的prefix和infix token IDs（保持不变）
llama_prefix = [128000, 128006, 882, 128007, 271]
llama_infix = [128009, 128006, 78191, 128007, 271]

# 将Llama-3-8B-Instruct的token IDs转换为文本（保持不变）
prefix_text = llama_8b_tokenizer.decode(llama_prefix)
infix_text = llama_8b_tokenizer.decode(llama_infix)

print("===== Llama-3-8B-Instruct格式 =====")
print("Prefix文本:", prefix_text)
print("Prefix Token IDs:", llama_prefix)
print("\nInfix文本:", infix_text)
print("Infix Token IDs:", llama_infix)

# 使用Llama-3.2-1B-Instruct的tokenizer将文本转换为对应的token IDs（更新模型关联）
llama_1b_prefix = llama_1b_tokenizer.encode(prefix_text, add_special_tokens=False)
llama_1b_infix = llama_1b_tokenizer.encode(infix_text, add_special_tokens=False)

print("\n===== Llama-3.2-1B-Instruct格式 =====")
print("Prefix文本:", prefix_text)
print("Llama-3.2-1B-Instruct Prefix Token IDs:", llama_1b_prefix)
print("\nInfix文本:", infix_text)
print("Llama-3.2-1B-Instruct Infix Token IDs:", llama_1b_infix)

# 显示Llama-3.2-1B-Instruct每个token的解码结果（更新模型关联）
print("\n===== Llama-3.2-1B-Instruct Prefix逐个token解码 =====")
for i, token_id in enumerate(llama_1b_prefix):
    token_text = llama_1b_tokenizer.decode([token_id])
    print(f"Token {i} (ID: {token_id}): '{token_text}'")

print("\n===== Llama-3.2-1B-Instruct Infix逐个token解码 =====")
for i, token_id in enumerate(llama_1b_infix):
    token_text = llama_1b_tokenizer.decode([token_id])
    print(f"Token {i} (ID: {token_id}): '{token_text}'")

