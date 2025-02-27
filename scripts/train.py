import torch
import os
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# ✅ 强制使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 让 GPU 参与计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 检查 GPU

# ✅ 2. 加载 GPT-2 Tiny 配置
config = GPT2Config(
    vocab_size=50257, n_positions=512, n_embd=128, n_layer=4, n_head=4
)
model = GPT2LMHeadModel(config).to(device)  # 🚀 确保模型在 GPU

# ✅ 启用 Gradient Checkpointing（减少显存占用，允许更大 batch_size）
model.gradient_checkpointing_enable()

# ✅ 3. 加载数据集（减少数据量，加快训练）
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:10%]")  # 只用 10% 数据，加速训练
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ 解决 padding 问题
tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS 作为 padding

# ✅ 4. Tokenization，优化 batch 加载
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)  # 🚀 降低 max_length
    tokens["labels"] = tokens["input_ids"].copy()  # 让 labels 和 input_ids 一致
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  # 确保 labels 被传入

# ✅ 使用 DataCollatorForLanguageModeling，支持不同 batch size
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ 5. 训练参数优化（加速训练）
training_args = TrainingArguments(
    output_dir="./models/checkpoint",
    per_device_train_batch_size=16,  # 🚀 提高 batch_size，加快训练
    gradient_accumulation_steps=4,  # 🚀 累积梯度，模拟更大 batch
    num_train_epochs=1,  # 🚀 降低 epochs，只训练 1 轮
    save_steps=5000,  # 🚀 降低 checkpoint 频率
    logging_steps=500,
    evaluation_strategy="no",
    learning_rate=5e-4,
    weight_decay=0.01,
    fp16=True,  # 🚀 启用混合精度，减少显存占用
    save_total_limit=1,  # 只保留最近的 checkpoint，节省磁盘
)

# ✅ 6. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator  # 🚀 确保 batch 适应不同长度
)

trainer.train()

# ✅ 7. 释放内存，防止 Colab 崩溃
torch.cuda.empty_cache()
import gc
gc.collect()
