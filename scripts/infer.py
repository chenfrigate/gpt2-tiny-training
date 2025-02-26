import torch
from transformers import GPT2LMHeadModel, AutoTokenizer

# 加载训练好的模型
model = GPT2LMHeadModel.from_pretrained("../models/final_model")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# 生成文本
input_text = "The future of AI is"
inputs = tokenizer(input_text, return_tensors="pt")
output = model.generate(**inputs, max_length=50)

print(tokenizer.decode(output[0]))
