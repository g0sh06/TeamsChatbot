from chatbot import *
from modifyText import text_dataset, tokenize_function
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


tokenized_dataset = text_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

train_test = text_dataset.train_test_split(test_size=0.1)
tokenized_train = train_test["train"].map(tokenize_function, batched=True)
tokenized_val = train_test["test"].map(tokenize_function, batched=True)

training_args = TrainingArguments(
    output_dir="./tinyllama-finetuned-cpu",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_ratio=0.1,
    num_train_epochs=5,
    learning_rate=3e-5,
    logging_dir="./logs",
    logging_steps=10,
    save_total_limit=1,
    no_cuda=True, 
    optim="adamw_torch",
    report_to="none",
    disable_tqdm=False,
    fp16=False,
    eval_steps=500,  
    save_steps=500, 
)

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    # Calculate perplexity
    loss = torch.nn.CrossEntropyLoss()(torch.tensor(preds), torch.tensor(labels))
    return {"perplexity": torch.exp(loss).item()}

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

label_names = ["input_ids"]  
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
    compute_metrics=compute_metrics
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
