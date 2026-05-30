import torch
from datasets import load_from_disk
from transformers import PreTrainedTokenizerFast, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model

if __name__ == '__main__':
    # Sử dụng Qwen2.5-1.5B
    MODEL_NAME = 'Qwen/Qwen2.5-0.5B-Instruct'

    tokenizer = PreTrainedTokenizerFast.from_pretrained('../bpe-tokenizer')
    tokenized_dataset = load_from_disk('../tokenized_dataset')

    dataset_splits = tokenized_dataset["train"].train_test_split(test_size=0.05, seed=42)

    train_data = dataset_splits["train"]
    eval_data = dataset_splits["test"]


    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
    )
    # Resize lại vocab của model với tokenizer BPE
    model.resize_token_embeddings(len(tokenizer))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # LoRA
    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, peft_config=peft_config)
    model.to(device)
    # model.print_trainable_parameters()
    training_args = TrainingArguments(
        output_dir="../output",
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        optim="adamw_torch_fused",
        fp16=not torch.cuda.is_bf16_supported(),
        logging_dir="../logs",
        logging_steps=1,
        push_to_hub=False
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        data_collator=data_collator,
    )

    print('------------TRAINING---------------')
    trainer.train()
    trainer.save_model("../fine_tuned_sales_model_final")
    tokenizer.save_pretrained("../fine_tuned_sales_model_final")