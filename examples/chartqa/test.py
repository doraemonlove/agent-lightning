from transformers import AutoTokenizer
import json
model_path = "/models/Qwen3-VL-8B-Instruct" 

try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    with open("../cua/triplets.json", "r") as f:
        data = json.load(f)
    prompt_token_ids = data[0]["prompt"]["token_ids"]
    response_token_ids = data[0]["response"]["token_ids"]
    text = tokenizer.decode(prompt_token_ids)
    print("\n\nprompt解码结果：")
    print(text)

    text = tokenizer.decode(response_token_ids)
    print("\n\nresponse解码结果：")
    print(text)

except Exception as e:
    print(f"加载 Tokenizer 失败: {e}")