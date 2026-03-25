import argparse
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

parser = argparse.ArgumentParser("train_teacher")
parser.add_argument("subset_idx", help="subset index.", type=int)
parser.add_argument("lr_exponent", help="exponent of lr.", type=float)
parser.add_argument("seed", help="random seed.", type=int)
parser.add_argument("is_distill_str", help="distill or not", type=str)

args = parser.parse_args()
subset_idx = args.subset_idx
lr_exponent = args.lr_exponent
seed = args.seed
is_distill_str = args.is_distill_str

is_distill = True if is_distill_str == "distill" else False

if is_distill:
    distillation_alpha = 0.0
    run_name = "gpt2-distill"
else:
    distillation_alpha = 1.0
    run_name = "gpt2"

learning_rate = 10 ** (-lr_exponent)
min_learning_rate = 10 ** (-(lr_exponent + 1))

print("subset id", subset_idx)

config = {
    "is_distill": is_distill,
    "subset": f"subset_{subset_idx:02d}",   # set which subset
    "total_subsets": "05",
    "train_dataset_path": f"/c4_train",
    "eval_dataset_path": f"/c4_train",
    "train_size": int(2.48e9 / 1024),

    "output_dir": f"/c4_model",
    "teacher_model_name": f"/c4_model/subset_{subset_idx:02d}/gpt2-medium-lr0.0005-seed57/checkpoint-11694",
    "student_model_name": "gpt2",

    "wandb_project": "c4",
    "wandb_entity": "",
    "run_name": run_name,

    "num_train_epochs": 1,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 1,
    "learning_rate": learning_rate,
    "weight_decay": 0.01,
    "warmup_ratio": 0.01,
    "lr_scheduler_type": "cosine_with_min_lr",
    "lr_scheduler_kwargs": {"min_lr": min_learning_rate},
    "gradient_accumulation_steps": 8,
    "max_grad_norm": 1.0,

    # loss = α * CE(student, labels) + (1−α) * KD_loss(student_logits, teacher_logits; T )
    "distillation_alpha": distillation_alpha,
    "distillation_temperature": 1.0,

    "logging_dir": f"/c4_gpt2_log",
    "logging_steps": 50,
    "eval_steps": 100,
    "save_steps": 1000,
    "save_total_limit": 3,

    "fp16": True,
    "dataloader_num_workers": 8,

    "seed": seed
}

set_seed(config["seed"])

def main():
    os.environ["WANDB_PROJECT"] = f"{config['wandb_project']}-{config['subset']}-{config['total_subsets']}"
    if config["wandb_entity"] is not None:
        os.environ["WANDB_ENTITY"] = config["wandb_entity"]

    accelerator = Accelerator()

    print("Starting GPT-2 training")

    print("Loading train dataset from", config["train_dataset_path"])
    train_dataset = load_from_disk(f"{config['train_dataset_path']}/{config['subset']}_train")
    train_dataset = train_dataset.shuffle(seed=config["seed"])
    train_dataset = train_dataset.select(range(config["train_size"]))

    print("Loading eval dataset from", config["eval_dataset_path"])
    eval_dataset = load_from_disk(f"{config['eval_dataset_path']}/{config['subset']}_eval")
    print("Train subset size: examples", len(train_dataset))
    print("Eval subset size: examples", len(eval_dataset))

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

    # This config is for GPT2-small:
    configuration = GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=1024,
        n_ctx=1024,
        n_embd=768,
        n_layer=12,
        n_head=12
    )

    student_model = GPT2LMHeadModel(configuration)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir=os.path.join(config["output_dir"], config["subset"], f"{config['run_name']}-lr{config['learning_rate']:.1e}-seed{config['seed']}"),
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

        logging_dir=os.path.join(config["logging_dir"], config["subset"]),
        logging_steps=config["logging_steps"],
        eval_strategy="steps",
        eval_steps=config["eval_steps"],
        save_strategy="steps",
        save_steps=config["save_steps"],
        save_total_limit=config["save_total_limit"],

        fp16=config["fp16"],

        dataloader_num_workers=config["dataloader_num_workers"],

        report_to="wandb",
        run_name=f"{config['run_name']}-lr{config['learning_rate']:.1e}-seed{config['seed']}",
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
