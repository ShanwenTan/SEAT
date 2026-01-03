from transformers import AutoTokenizer

# 加载Mistral-7B-Instruct模型的tokenizer
mistral_model_path = "/data/saffron/local_model_cache/shakechen/Llama-2-7b-chat-hf"
mistral_tokenizer = AutoTokenizer.from_pretrained(mistral_model_path)

# LLaMA-3的prefix文本
llama_prefix_text = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n"

# 将文本转换为Mistral模型的token IDs
mistral_prefix_ids = mistral_tokenizer.encode(llama_prefix_text, add_special_tokens=False)

print("LLaMA-3 Prefix文本:", llama_prefix_text)
print("Mistral-7B-Instruct Prefix Token IDs:", mistral_prefix_ids)

# 可选：验证转换是否正确
print("\n转换验证 - 将Mistral的Token IDs解码回文本:")
decoded_text = mistral_tokenizer.decode(mistral_prefix_ids)
print("解码结果:", repr(decoded_text))