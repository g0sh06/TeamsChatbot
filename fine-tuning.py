from chatbot import *
from modifyText import text_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import Dataset

tokenized_dataset = text_dataset.map(
    lambda examples: tokenizer(examples["text"], truncation=True, max_length=512),
    batched=True,
    remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    warmup_steps=10,
    num_train_epochs=3,
    learning_rate=2e-4,
    fp16=True,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=2,
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
print("Sample tokenized example:", tokenized_dataset[0])

label_names = ["input_ids"]  
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

trainer.train()

model.save_pretrained("my_tinyllama_finetuned")
tokenizer.save_pretrained("my_tinyllama_finetuned")
