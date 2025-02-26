import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer
from datasets import load_dataset

# 加载 GPT-2 Tiny 配置
config = GPT2Config(
    vocab_size=50257, n_positions=512, n_embd=128, n_layer=4, n_head=4
)
model = GPT2LMHeadModel(config)

# 加载数据
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenized_data = tokenizer(dataset['train']['text'], truncation=True, padding=True, max_length=512, return_tensors="pt")

# 训练参数
training_args = TrainingArguments(
    output_dir="../models/checkpoint",
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_steps=1000,
    evaluation_strategy="epoch",
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True
)

# 训练模型
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data
)

trainer.train()
