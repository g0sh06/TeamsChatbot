from chatbot import *
from modifyText import tokenized_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

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
    eval_strategy="no",
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
