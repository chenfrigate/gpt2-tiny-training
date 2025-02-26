import torch
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer, DataCollatorWithPadding
from datasets import load_dataset

# 🚀 1. 确保 GPU 设备可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 输出 GPU 信息

# 🚀 2. 加载 GPT-2 Tiny 配置
config = GPT2Config(
    vocab_size=50257, n_positions=512, n_embd=128, n_layer=4, n_head=4
)
model = GPT2LMHeadModel(config).to(device)  # 🔥 将模型移动到 GPU

# 🚀 3. 加载数据集
dataset = load_dataset("wikitext", "wikitext-103-raw-v1")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ 解决 padding 问题
tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS 作为 padding

# 🚀 4. Tokenization 并确保所有序列长度一致
def tokenize_function(examples):
    return tokenizer(examples['text'], 
                     truncation=True, 
                     padding="max_length",  # ✅ 统一填充到 max_length
                     max_length=512)

tokenized_dataset = dataset.map(tokenize_function, batched=True)

# ✅ 确保数据格式正确（转换为 PyTorch 张量）
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
tokenized_dataset = tokenized_dataset["train"]  # 只取训练集

# ✅ 使用 DataCollator 让 batch 适应不同长度
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

# 🚀 5. 训练参数，优化显存占用
training_args = TrainingArguments(
    output_dir="./models/checkpoint",
    per_device_train_batch_size=4,  # 🔥 降低 batch_size 避免 OOM
    num_train_epochs=3,
    save_steps=1000,
    logging_steps=500,
    evaluation_strategy="no",
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,  # 🔥 启用混合精度训练，加速并降低显存占用
    save_total_limit=2
)

# 🚀 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator  # ✅ 让 batch 适应不同长度
)

trainer.train()

# 🚀 7. 释放内存，防止 Colab 崩溃
torch.cuda.empty_cache()
import gc
gc.collect()
