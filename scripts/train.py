import torch
import os
from transformers import GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, AutoTokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset

# ✅ 强制使用 GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # 让 GPU 参与计算
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")  # 检查 GPU

# ✅ 1. 加载 GPT-2 Tiny 配置
config = GPT2Config(
    vocab_size=50257, n_positions=512, n_embd=128, n_layer=4, n_head=4
)
model = GPT2LMHeadModel(config).to(device)  # 🚀 确保模型在 GPU

# ✅ 2. 启用 Gradient Checkpointing（减少显存占用，允许更大 batch_size）
use_gradient_checkpointing = False  # 🚀 设为 False 以提高计算效率（如果显存足够）
if use_gradient_checkpointing:
    model.gradient_checkpointing_enable()
else:
    model.gradient_checkpointing_disable()

# ✅ 3. 加载数据集（增加数据量，提高训练效果）
dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train[:80%]")  # 🚀 增加数据量
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# ✅ 4. 解决 padding 问题
tokenizer.pad_token = tokenizer.eos_token  # 使用 EOS 作为 padding

# ✅ 5. Tokenization，优化 batch 加载
def tokenize_function(examples):
    tokens = tokenizer(examples['text'], truncation=True, padding="max_length", max_length=256)
    tokens["labels"] = tokens["input_ids"].copy()  # ✅ 让 labels 和 input_ids 一致
    return tokens

tokenized_dataset = dataset.map(tokenize_function, batched=True)
tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

# ✅ 6. 使用 DataCollatorForLanguageModeling，支持不同 batch size
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ✅ 7. 训练参数优化（确保可恢复）
training_args = TrainingArguments(
    output_dir="./models/checkpoint",
    per_device_train_batch_size=32,  # 🚀 提高 batch_size，加快训练
    gradient_accumulation_steps=1,  # 🚀 累积梯度，模拟更大 batch
    num_train_epochs=3,  # 🚀 训练 3 轮
    save_steps=1000,  # ✅ 增加 checkpoint 频率，防止训练丢失
    logging_steps=500,
    evaluation_strategy="no",  # ✅ 关闭自动评估
    eval_steps=None,  # ✅ 避免 Trainer 误调用 eval
    learning_rate=2e-4,
    weight_decay=0.01,
    fp16=True,  # 🚀 启用混合精度，减少显存占用
    save_total_limit=2,  # ✅ 只保留最近 2 个 checkpoint，防止占用磁盘
    save_strategy="steps",  # ✅ 仍然按 `steps` 方式保存 checkpoint
    load_best_model_at_end=False,  # ✅ 关闭最优模型加载
)

# ✅ 8. 检查是否有 `checkpoint` 进行恢复训练
checkpoint_dir = "./models/checkpoint"
resume_from_checkpoint = None

if os.path.exists(checkpoint_dir) and len(os.listdir(checkpoint_dir)) > 0:
    latest_checkpoint = max(
        [os.path.join(checkpoint_dir, d) for d in os.listdir(checkpoint_dir) if "checkpoint-" in d],
        key=os.path.getmtime
    )
    resume_from_checkpoint = latest_checkpoint
    print(f"Resuming from checkpoint: {resume_from_checkpoint}")
else:
    print("No checkpoint found, starting fresh training...")

# ✅ 9. 训练
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train(resume_from_checkpoint=resume_from_checkpoint)  # ✅ 自动恢复

# ✅ 10. 释放内存，防止 Colab 崩溃
torch.cuda.empty_cache()
import gc
gc.collect()
