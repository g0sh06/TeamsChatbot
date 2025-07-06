from chatbot import *
from modifyText import tokenized_dataset
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling

# Training arguments
training_args = TrainingArguments(
    per_device_train_batch_size = 4,
    gradient_accumulation_steps = 4,
    learning_rate = 1e-3,
    num_train_epochs = 50,
    fp16 = True,
    logging_steps = 20,
    save_strategy = 'epoch',
    report_to = 'none',
    remove_unused_columns = False,      
    label_names = ["labels"]
)

model.enable_input_require_grads()
model.gradient_checkpointing_enable()

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
    tokenizer=tokenizer,
)

try:
    print("\n=== Starting Training ===")
    train_results = trainer.train()
    
    # Save with proper formatting
    trainer.save_model("my_tinyllama_finetuned")
    tokenizer.save_pretrained("my_tinyllama_finetuned")
    
    print("\n=== Training Complete ===")
    print(f"Final loss: {train_results.training_loss:.4f}")
    
except Exception as e:
    print(f"\nTraining failed: {str(e)}")
    if "memory" in str(e).lower():
        print("Try: 1) Reduce batch size 2) Decrease chunk_size 3) Use gradient checkpointing")
