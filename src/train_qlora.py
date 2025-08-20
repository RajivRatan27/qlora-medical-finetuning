# train_qlora.py
import os, math, json, argparse
import torch
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from collator import SFTCollator

def build_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_id", type=str, required=True,
                    help="e.g. meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--train_file", type=str, required=True)
    ap.add_argument("--eval_file", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs")

    # ==== Paper hyperparams & toggles ====
    # Algorithm 2: batch=4, grad_accum=8, epochs=10, log=50, save=500. :contentReference[oaicite:7]{index=7}
    ap.add_argument("--per_device_train_batch_size", type=int, default=4)
    ap.add_argument("--per_device_eval_batch_size", type=int, default=4)
    ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
    ap.add_argument("--num_train_epochs", type=int, default=10)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=500)

    # LR options from paper:
    #   (a) Algorithm 2: 2e-5 AdamW
    #   (b) §4.3.1 two-stage: 5e-6 and 1e-4 depending on epoch. 
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Used when --two_stage_lr is disabled.")
    ap.add_argument("--two_stage_lr", action="store_true",
                    help="Enable 2-phase LR schedule: 5e-6 then 1e-4 (per paper §4.3.1).")

    # QLoRA / LoRA from Algorithm 1: 4-bit, fp16 compute; r, alpha, dropout; target Q/V. :contentReference[oaicite:9]{index=9}
    ap.add_argument("--lora_r", type=int, default=16,  # §4.3.1 fixed at 16; Alg.1 shows 64. Pick via flag. 
                    help="Use 16 to match §4.3.1, or 64 to match Algorithm 1.")
    ap.add_argument("--lora_alpha", type=int, default=16)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--target_modules", type=str, nargs="+", default=["q_proj","v_proj"],
                    help="Alg.1 targets query/value projections.")
    ap.add_argument("--max_length", type=int, default=2048)

    # Misc
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--fp16", action="store_true", help="Mixed precision fp16.")
    return ap.parse_args()

def build_scheduler(optimizer, num_training_steps, two_stage=False):
    if not two_stage:
        # Transformers Trainer will create a scheduler automatically (linear w/ warmup=0)
        return None

    # Two-stage LR: first half 5e-6, second half 1e-4 (from §4.3.1). :contentReference[oaicite:11]{index=11}
    from torch.optim.lr_scheduler import LambdaLR
    lr1, lr2 = 5e-6, 1e-4
    def lr_lambda(step):
        ratio = step / max(1, num_training_steps)
        return (lr1 if ratio < 0.5 else lr2) / lr1
    return LambdaLR(optimizer, lr_lambda)

def main():
    args = build_args()
    torch.manual_seed(args.seed)

    # ---- Dataset (JSONL) ----
    data_files = {"train": args.train_file, "validation": args.eval_file}
    ds = load_dataset("json", data_files=data_files)

    # ---- Tokenizer ----
    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # ---- 4-bit quantization (QLoRA) ----  (Alg.1 uses 4-bit + fp16 compute) :contentReference[oaicite:12]{index=12}
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,  # fp16 compute per Alg.1
        bnb_4bit_quant_type="nf4",             # standard QLoRA choice
        bnb_4bit_use_double_quant=True
    )
    base = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        quantization_config=bnb_cfg,
        device_map="auto",
        trust_remote_code=True
    )
    base = prepare_model_for_kbit_training(base)

    # ---- LoRA adapters ----
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=args.target_modules,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(base, lora_cfg)

    # ---- Collator ----
    collator = SFTCollator(tok, max_length=args.max_length)

    # ---- TrainingArguments (paper defaults) ----
    train_args = TrainingArguments(
        output_dir=args.out_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.lr,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        evaluation_strategy="steps",
        eval_steps=args.save_steps,
        save_total_limit=2,
        bf16=not args.fp16 and torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8,
        fp16=args.fp16,
        report_to="none",
        lr_scheduler_type="linear",  # replaced by custom scheduler if two_stage_lr
        optim="adamw_torch",         # AdamW per Algorithm 2
        remove_unused_columns=False
    )

    # ---- Trainer ----
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tok,
        data_collator=collator
    )

    # Optional two-stage LR from §4.3.1 (5e-6 → 1e-4). :contentReference[oaicite:13]{index=13}
    if args.two_stage_lr:
        trainer.create_optimizer_and_scheduler(num_training_steps=trainer.get_num_train_steps())
        trainer.lr_scheduler = build_scheduler(trainer.optimizer, trainer.state.max_steps, two_stage=True)

    # ---- Train ----
    trainer.train()

    # ---- Save adapters only (PEFT) ----
    trainer.model.save_pretrained(os.path.join(args.out_dir, "adapter"))
    tok.save_pretrained(args.out_dir)

if __name__ == "__main__":
    main()
