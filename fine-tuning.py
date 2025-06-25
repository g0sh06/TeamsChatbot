from chatbot import *
from modifyText import text_dataset, tokenize_function
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


tokenized_dataset = text_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned-cpu",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    warmup_steps=50,
    num_train_epochs=3,
    learning_rate=2e-5,
    logging_dir="./logs",
    logging_steps=5,
    save_total_limit=1,
    no_cuda=True, 
    optim="adamw_torch",  
    report_to="none",
    disable_tqdm=False
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

label_names = ["input_ids"]  
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

try:
    trainer.train()
    model.save_pretrained("my_tinyllama_finetuned")
    tokenizer.save_pretrained("my_tinyllama_finetuned")
    print("Training completed successfully!")
except Exception as e:
    print(f"Training failed: {e}")
    if "CUDA out of memory" in str(e):
        print("Try reducing batch size or sequence length")
