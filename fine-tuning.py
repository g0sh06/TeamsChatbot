from chatbot import *
from modifyText import text_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

# Tokenize and prepare dataset
tokenized_dataset = text_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# For causal LM, labels should be same as input_ids
tokenized_dataset = tokenized_dataset.map(
    lambda examples: {"labels": examples["input_ids"].copy()},
    batched=True
)

# Training arguments
training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=5,  # Increased epochs
    learning_rate=1e-5,  # Lower learning rate
    warmup_ratio=0.1,
    weight_decay=0.01,
    optim="adamw_torch",
    logging_steps=10,
    save_steps=200,
    evaluation_strategy="no",
    save_total_limit=1,
    use_cpu=True,
    report_to="none",
    remove_unused_columns=False
)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
model.save_pretrained("my_tinyllama_finetuned")
tokenizer.save_pretrained("my_tinyllama_finetuned")
