from accelerate import Accelerator
import os
import wandb
from transformers import (
    GPT2Tokenizer,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    GPT2Config,
    GPT2LMHeadModel,
    set_seed
)
from datasets import load_from_disk
from distill_trainer import DistillationTrainer

config = {
    "is_distill": False, # modify this
    "train_dataset_path": "/c4_train_teacher_7.1B",
    "eval_dataset_path": "/c4_eval_indexed",

    "output_dir": "/c4_gpt2_model",
    "teacher_model_name": "",
    "student_model_name": "gpt2",

    "wandb_project": "gpt2-c4-entropy",
    "wandb_entity": "",
    "run_name": "gpt2-medium", # and this

    "num_train_epochs": 1,
    "per_device_train_batch_size": 8,
    "per_device_eval_batch_size": 1,
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "lr_scheduler_type": "cosine_with_min_lr",
    "lr_scheduler_kwargs": {"min_lr": 5e-5},
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,

    # loss = α * CE(student, labels) + (1−α) * KD_loss( student_logits, teacher_logits; T )
    "distillation_alpha": 1.0,
    "distillation_temperature": 1.0,

    "logging_dir": "/c4_gpt2_log",
    "logging_steps": 100,
    "eval_steps": 250,
    "save_steps": 1000,
    "save_total_limit": 2,

    "fp16": True,
    "dataloader_num_workers": 8,

    "seed": 57
}

set_seed(config["seed"])

def main():
    os.environ["WANDB_PROJECT"] = f"{config['wandb_project']}"
    if config["wandb_entity"] is not None:
        os.environ["WANDB_ENTITY"] = config["wandb_entity"]

    accelerator = Accelerator()

    print("Starting GPT-2 training")

    print("Loading train dataset from", config["train_dataset_path"])
    train_dataset = load_from_disk(f"{config['train_dataset_path']}")
    print("Loading eval dataset  from", config["eval_dataset_path"])
    eval_dataset = load_from_disk(f"{config['eval_dataset_path']}")
    print("Train split size:", len(train_dataset))
    print("Eval split size:", len(eval_dataset))

    tokenizer = GPT2Tokenizer.from_pretrained(config["student_model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if config["is_distill"]:
        print("Loading teacher model:", config["teacher_model_name"])
        teacher_model = GPT2LMHeadModel.from_pretrained(config["teacher_model_name"])
        teacher_model.eval()
        teacher_model.to(accelerator.device)
    else:
        print("No distillation, so no teacher model")
        teacher_model = None

    print("Loading student model:", config["student_model_name"])

    # This config is for GPT2-medium:
    configuration = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=1024,
        n_head=16,
        n_layer=24,
    )

    student_model = GPT2LMHeadModel(configuration)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], f"{config['run_name']}-lr{config['learning_rate']}-seed{config['seed']}"),
        overwrite_output_dir=True,

        num_train_epochs=config["num_train_epochs"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config["per_device_eval_batch_size"],
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        warmup_ratio=config["warmup_ratio"],
        lr_scheduler_type=config["lr_scheduler_type"],
        lr_scheduler_kwargs=config["lr_scheduler_kwargs"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        max_grad_norm=config["max_grad_norm"],

        logging_dir=config["logging_dir"],
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],

        fp16=config["fp16"],

        dataloader_num_workers=config["dataloader_num_workers"],

        report_to="wandb",
        run_name=f"{config['run_name']}-lr{config['learning_rate']}-seed{config['seed']}",
    )

    print("Initializing DistillationTrainer")
    trainer = DistillationTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=training_args,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,

        tokenizer=tokenizer,

        alpha=config["distillation_alpha"],
        temperature=config["distillation_temperature"],
    )

    print("***** Starting training *****")
    
    # eval first
    metrics = trainer.evaluate()
    print(metrics)

    trainer.train()
    print("***** Training finished; saving final model *****")
    trainer.save_model(config["output_dir"])

    wandb.finish()

if __name__ == "__main__":
    main()