import argparse
import os
from pathlib import Path

from loguru import logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
import inspect
from datasets import load_dataset
import math


def build_lora_config(
    r: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    target_modules=None,
):
    if target_modules is None:
        target_modules = [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        task_type=TaskType.CAUSAL_LM,
    )


def parse_args():
    p = argparse.ArgumentParser(description="LoRA finetune OuteTTS-1.0-0.6B with v3 prompts")
    p.add_argument("--model_path", default="OuteTTS-1.0-0.6B", help="Base model local dir or repo id")
    p.add_argument("--tokenizer_path", default="OuteTTS-1.0-0.6B", help="Tokenizer local dir or repo id")
    p.add_argument("--train_jsonl", default="lsy/training_data_v3.jsonl", help="v3 training JSONL file")
    p.add_argument("--output_dir", default="outetts_finetuned_v3_100", help="Output directory for LoRA adapter")
    p.add_argument("--max_seq_length", type=int, default=2048, help="Max sequence length for training")
    p.add_argument("--batch_size", type=int, default=1, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=16, help="Gradient accumulation steps")
    p.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    p.add_argument("--epochs", type=float, default=3.0, help="Number of epochs")
    p.add_argument("--save_steps", type=int, default=500, help="Save every N steps")
    p.add_argument("--logging_steps", type=int, default=10, help="Log every N steps")
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 training")
    p.add_argument("--fp16", action="store_true", help="Use float16 training")
    p.add_argument("--warmup_steps", type=int, default=100, help="Warmup steps")
    p.add_argument("--lora_r", type=int, default=16, help="LoRA rank r")
    p.add_argument("--lora_alpha", type=int, default=32, help="LoRA alpha")
    p.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    # Evaluation / logging controls
    p.add_argument("--eval_jsonl", help="Validation JSONL file; if omitted, optional split from train is used")
    p.add_argument("--eval_split_ratio", type=float, default=0.0, help="Hold-out ratio from train when eval_jsonl is not provided (0.0 to disable)")
    p.add_argument("--eval_steps", type=int, default=200, help="Run evaluation every N steps when eval dataset is available")
    p.add_argument("--save_total_limit", type=int, default=2, help="Limit total checkpoints to keep")
    p.add_argument("--load_best_at_end", action="store_true", help="Load best checkpoint at end based on eval loss")
    p.add_argument("--early_stopping_patience", type=int, default=0, help="Enable early stopping with this patience (0 to disable)")
    p.add_argument("--early_stopping_delta", type=float, default=0.0, help="Minimum improvement to reset patience (used by EarlyStoppingCallback)")
    p.add_argument("--eval_before_train", action="store_true", help="Run a single evaluation pass before training starts (requires eval dataset)")
    # Optional advanced knobs
    p.add_argument("--lr_scheduler_type", default="cosine", help="LR scheduler type (e.g., cosine, linear)")
    p.add_argument("--packing", action="store_true", help="Enable sequence packing if your dataset supports it")
    p.add_argument("--dataset_text_field", default="text", help="Field name in dataset JSONL for text")
    p.add_argument("--target_modules", nargs='*', default=None, help="LoRA target module names (space-separated)")
    # Hyperparameters via JSON
    p.add_argument("--hparams", help="Path to JSON with hyperparameters to load and override CLI defaults")
    p.add_argument("--save_hparams", help="Path to save resolved hyperparameters as JSON for future runs")
    p.add_argument("--init_hparams", help="Write a template hyperparameters JSON and exit")
    p.add_argument(
        "--force_local",
        action="store_true",
        help="Force offline mode and load only from local files",
    )
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Write a template hparams file and exit (no training)
    if args.init_hparams:
        import json as _json
        keys = [
            "model_path", "tokenizer_path", "train_jsonl", "output_dir",
            "max_seq_length", "batch_size", "grad_accum", "lr", "epochs",
            "save_steps", "logging_steps", "bf16", "fp16", "warmup_steps",
            "lora_r", "lora_alpha", "lora_dropout", "seed", "lr_scheduler_type",
            "packing", "dataset_text_field", "target_modules", "force_local",
            # evaluation / early stopping knobs
            "eval_jsonl", "eval_split_ratio", "eval_steps", "save_total_limit",
            "load_best_at_end", "early_stopping_patience", "early_stopping_delta", "eval_before_train",
        ]
        payload = {k: getattr(args, k, None) for k in keys}
        with open(args.init_hparams, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.success(f"Initialized hyperparameters template at: {args.init_hparams}")
        return

    # Load hyperparameters from JSON (if provided) and override args
    if args.hparams:
        import json as _json
        hp_path = Path(args.hparams)
        if not hp_path.exists():
            raise FileNotFoundError(f"Hyperparameter file not found: {hp_path}")
        with open(hp_path, "r", encoding="utf-8") as f:
            hp = _json.load(f)
        unknown = []
        for k, v in hp.items():
            if hasattr(args, k):
                setattr(args, k, v)
            else:
                unknown.append(k)
        if unknown:
            logger.warning(f"Ignoring unknown hparams keys: {unknown}")

    # Optionally save resolved hyperparameters for reproducibility
    if args.save_hparams:
        import json as _json
        keys = [
            "model_path", "tokenizer_path", "train_jsonl", "output_dir",
            "max_seq_length", "batch_size", "grad_accum", "lr", "epochs",
            "save_steps", "logging_steps", "bf16", "fp16", "warmup_steps",
            "lora_r", "lora_alpha", "lora_dropout", "seed", "lr_scheduler_type",
            "packing", "dataset_text_field", "target_modules", "force_local",
            # evaluation / early stopping knobs
            "eval_jsonl", "eval_split_ratio", "eval_steps", "save_total_limit",
            "load_best_at_end", "early_stopping_patience", "early_stopping_delta", "eval_before_train",
        ]
        payload = {k: getattr(args, k, None) for k in keys}
        with open(args.save_hparams, "w", encoding="utf-8") as f:
            _json.dump(payload, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved resolved hyperparameters to: {args.save_hparams}")

    # Prefer local directories; optionally force offline
    model_is_dir = Path(args.model_path).is_dir()
    tok_is_dir = Path(args.tokenizer_path).is_dir()

    # Fail fast if a path-like value doesn't resolve (often due to accidental line breaks)
    def _looks_like_path(p: str) -> bool:
        return any(s in p for s in (os.sep, "/", "\\")) or p.startswith(".")

    if _looks_like_path(args.model_path) and not model_is_dir:
        logger.error(
            "Model path does not exist: {}. If you broke the command across lines, put the entire path on one line or quote it (e.g., \"./OuteTTS-1.0-0.6B\").".format(
                args.model_path
            )
        )
        raise FileNotFoundError(args.model_path)
    if _looks_like_path(args.tokenizer_path) and not tok_is_dir:
        logger.error(
            "Tokenizer path does not exist: {}. Ensure it is a single, uninterrupted argument or quote it.".format(
                args.tokenizer_path
            )
        )
        raise FileNotFoundError(args.tokenizer_path)
    if args.force_local or model_is_dir or tok_is_dir:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    if model_is_dir and not (Path(args.model_path) / "config.json").exists():
        logger.error(f"Local model dir missing config.json: {args.model_path}")
        raise FileNotFoundError(f"Expected model files under: {args.model_path}")
    if tok_is_dir and not (Path(args.tokenizer_path) / "tokenizer_config.json").exists():
        logger.warning("Tokenizer dir missing tokenizer_config.json; attempting load anyway")

    logger.info(
        f"Loading base model from {'local dir' if model_is_dir else 'repo id'}: {args.model_path}"
    )

    torch_dtype = None
    if args.bf16:
        torch_dtype = torch.bfloat16
    elif args.fp16:
        torch_dtype = torch.float16

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch_dtype,
        device_map="auto",
        local_files_only=True if (args.force_local or model_is_dir) else False,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        local_files_only=True if (args.force_local or tok_is_dir) else False,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    lora_config = build_lora_config(
        r=args.lora_r, alpha=args.lora_alpha, dropout=args.lora_dropout,
        target_modules=args.target_modules if args.target_modules else None,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info(f"Loading dataset: {args.train_jsonl}")
    full_train = load_dataset("json", data_files=args.train_jsonl, split="train")
    eval_dataset = None
    train_dataset = full_train
    if args.eval_jsonl:
        logger.info(f"Loading eval dataset: {args.eval_jsonl}")
        eval_dataset = load_dataset("json", data_files=args.eval_jsonl, split="train")
    elif args.eval_split_ratio and args.eval_split_ratio > 0.0:
        logger.info(f"Splitting train/eval with ratio={args.eval_split_ratio}")
        split = full_train.train_test_split(test_size=args.eval_split_ratio, seed=args.seed)
        train_dataset, eval_dataset = split["train"], split["test"]

    # Log step estimates
    world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
    eff_batch = max(1, args.batch_size) * max(1, args.grad_accum) * max(1, world_size)
    steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
    total_steps = int(steps_per_epoch * args.epochs)
    logger.info(f"Samples: train={len(train_dataset)}{' | eval='+str(len(eval_dataset)) if eval_dataset is not None else ''}")
    logger.info(f"Devices: {world_size} | effective_batch={eff_batch} | steps_per_epoch={steps_per_epoch} | total_steps≈{total_steps}")

    # Build config kwargs and drop those unsupported by current TRL version
    _sft_cfg_kwargs = dict(
        output_dir=args.output_dir,
        dataset_text_field=args.dataset_text_field,
        max_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        bf16=args.bf16,
        fp16=args.fp16,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        seed=args.seed,
        packing=bool(args.packing),
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=(args.load_best_at_end if eval_dataset is not None else False),
        evaluation_strategy=("steps" if eval_dataset is not None else "no"),
        eval_steps=(args.eval_steps if eval_dataset is not None else None),
        save_strategy="steps",
        metric_for_best_model="loss",
        greater_is_better=False,
    )
    _allowed = set(inspect.signature(SFTConfig.__init__).parameters.keys())

    # If this TRL/Transformers combo does not support evaluation_strategy/save_strategy,
    # force-disable load_best_model_at_end to avoid post_init validation errors.
    if ("evaluation_strategy" not in _allowed) or ("save_strategy" not in _allowed) or (eval_dataset is None):
        if _sft_cfg_kwargs.get("load_best_model_at_end"):
            logger.warning(
                "Disabling load_best_model_at_end because evaluation/save strategy is unsupported or no eval dataset is provided."
            )
        _sft_cfg_kwargs["load_best_model_at_end"] = False
        # Drop eval-related keys proactively if unsupported
        for k in ["evaluation_strategy", "eval_steps", "save_strategy", "metric_for_best_model", "greater_is_better"]:
            _sft_cfg_kwargs.pop(k, None)

    _unsupported = [k for k in _sft_cfg_kwargs.keys() if k not in _allowed]
    if _unsupported:
        logger.warning(f"SFTConfig in current TRL version does not support: {_unsupported}. These options will be ignored.")
    _filtered = {k: v for k, v in _sft_cfg_kwargs.items() if k in _allowed}
    training_args = SFTConfig(**_filtered)

    # TRL compatibility: newer versions use `processing_class` instead of `tokenizer`
    trainer_init = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        **({"eval_dataset": eval_dataset} if eval_dataset is not None else {}),
    }
    sig = inspect.signature(SFTTrainer.__init__)
    if "processing_class" in sig.parameters:
        trainer = SFTTrainer(processing_class=tokenizer, **trainer_init)
    elif "tokenizer" in sig.parameters:
        trainer = SFTTrainer(tokenizer=tokenizer, **trainer_init)
    else:
        # Fallback: pass nothing; rely on model's internal tokenizer (may fail)
        trainer = SFTTrainer(**trainer_init)

    # Add a printer callback to always log eval metrics when evaluation runs
    class _EvalPrinter(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is not None:
                eval_loss = metrics.get("eval_loss")
                logger.info(f"[Eval] step={state.global_step} eval_loss={eval_loss} metrics={metrics}")
            return control

    trainer.add_callback(_EvalPrinter())

    # Optional early stopping (only when metrics/eval strategy are supported)
    if args.early_stopping_patience and eval_dataset is not None:
        metric = getattr(training_args, "metric_for_best_model", None)
        eval_strategy = getattr(training_args, "evaluation_strategy", None)
        if metric is None or (eval_strategy in (None, "no")):
            logger.warning(
                "Early stopping disabled: metric_for_best_model/evaluation_strategy not available in current TRL/Transformers."
            )
        else:
            trainer.add_callback(EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_delta,
            ))

    # Optional: run a single evaluation pass before training begins
    if args.eval_before_train and 'eval_dataset' in trainer.__dict__ and trainer.eval_dataset is not None:
        logger.info("Pre-training evaluation enabled. Running evaluation before training…")
        try:
            pre_metrics = trainer.evaluate()
            pre_eval_loss = pre_metrics.get('eval_loss', None)
            logger.info(f"Pre-train eval metrics: eval_loss={pre_eval_loss}, details={pre_metrics}")
            # Save to file
            try:
                import json as _json
                pre_metrics_path = Path(args.output_dir) / "eval_metrics_before.json"
                with open(pre_metrics_path, "w", encoding="utf-8") as f:
                    _json.dump(pre_metrics, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved pre-train eval metrics to: {pre_metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to save pre-train eval metrics: {e}")
        except Exception as e:
            logger.warning(f"Pre-training evaluation failed: {e}")

    logger.info("Starting training...")
    trainer.train()

    logger.success("Training finished. Saving adapter...")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info(f"Saved to: {args.output_dir}")

    # Evaluate after training and print/save results (if eval dataset available)
    if 'eval_dataset' in trainer.__dict__ and trainer.eval_dataset is not None:
        logger.info("Running evaluation at end of training…")
        try:
            metrics = trainer.evaluate()
            # Pretty-print key results
            eval_loss = metrics.get('eval_loss', None)
            logger.info(f"Eval metrics: eval_loss={eval_loss}, details={metrics}")
            # Persist metrics
            try:
                import json as _json
                metrics_path = Path(args.output_dir) / "eval_metrics_final.json"
                with open(metrics_path, "w", encoding="utf-8") as f:
                    _json.dump(metrics, f, indent=2, ensure_ascii=False)
                logger.info(f"Saved eval metrics to: {metrics_path}")
            except Exception as e:
                logger.warning(f"Failed to save eval metrics: {e}")
        except Exception as e:
            logger.warning(f"Evaluation after training failed: {e}")


if __name__ == "__main__":
    main()
