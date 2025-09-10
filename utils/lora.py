import os
import inspect
import math
from pathlib import Path
from types import SimpleNamespace

from loguru import logger

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, EarlyStoppingCallback
from transformers.trainer_callback import TrainerCallback
from peft import get_peft_model, LoraConfig, TaskType
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset


class LoraFinetuner:
    """Class-based LoRA finetuning for OuteTTS v3 prompts.

    Usage:
      from lora import LoraFinetuner
      LoraFinetuner.from_hparams('/path/to/lora_hparams.json').run()
    """

    DEFAULTS = {
        "model_path": "OuteTTS-1.0-0.6B",
        "tokenizer_path": "OuteTTS-1.0-0.6B",
        "train_jsonl": "wavs/training_data_v3.jsonl",
        "eval_jsonl": None,
        "output_dir": "outetts_finetuned_v3",
        "max_seq_length": 2048,
        "batch_size": 1,
        "grad_accum": 16,
        "lr": 2e-5,
        "epochs": 3.0,
        "save_steps": 500,
        "logging_steps": 10,
        "bf16": False,
        "fp16": False,
        "warmup_steps": 100,
        "lora_r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "seed": 42,
        "lr_scheduler_type": "cosine",
        "packing": False,
        "dataset_text_field": "text",
        "target_modules": None,
        "force_local": True,
        "eval_split_ratio": 0.0,
        "eval_steps": 200,
        "save_total_limit": 2,
        "load_best_at_end": True,
        "early_stopping_patience": 0,
        "early_stopping_delta": 0.0,
        "eval_before_train": False,
    }

    @staticmethod
    def build_lora_config(r=16, alpha=32, dropout=0.05, target_modules=None):
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

    def __init__(self, config: dict):
        cfg = dict(self.DEFAULTS)
        cfg.update({k: v for k, v in (config or {}).items() if v is not None})
        self.cfg = SimpleNamespace(**cfg)

    @classmethod
    def from_hparams(cls, path: str):
        import json as _json
        hp_path = Path(path)
        if not hp_path.exists():
            raise FileNotFoundError(f"Hyperparameter file not found: {hp_path}")
        with open(hp_path, "r", encoding="utf-8") as f:
            cfg = _json.load(f)
        # Default output_dir to sibling of hparams if not absolute
        if not cfg.get("output_dir"):
            cfg["output_dir"] = str(hp_path.parent / "outetts_finetuned_v3")
        return cls(cfg)

    def _prepare_env(self):
        model_is_dir = Path(self.cfg.model_path).is_dir()
        tok_is_dir = Path(self.cfg.tokenizer_path).is_dir()
        # Fail fast for path-like strings that do not exist
        def _looks_like_path(p: str) -> bool:
            return any(s in p for s in (os.sep, "/", "\\")) or p.startswith(".")
        if _looks_like_path(self.cfg.model_path) and not model_is_dir:
            raise FileNotFoundError(self.cfg.model_path)
        if _looks_like_path(self.cfg.tokenizer_path) and not tok_is_dir:
            raise FileNotFoundError(self.cfg.tokenizer_path)
        if self.cfg.force_local or model_is_dir or tok_is_dir:
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        if model_is_dir and not (Path(self.cfg.model_path) / "config.json").exists():
            raise FileNotFoundError(f"Expected model files under: {self.cfg.model_path}")

    def _load_model_tokenizer(self):
        torch_dtype = None
        if self.cfg.bf16:
            torch_dtype = torch.bfloat16
        elif self.cfg.fp16:
            torch_dtype = torch.float16
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            local_files_only=bool(self.cfg.force_local),
        )
        tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.tokenizer_path,
            local_files_only=bool(self.cfg.force_local),
            use_fast=True,
        )
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return model, tokenizer

    def _prepare_datasets(self):
        logger.info(f"Loading dataset: {self.cfg.train_jsonl}")
        full_train = load_dataset("json", data_files=self.cfg.train_jsonl, split="train")
        eval_dataset = None
        train_dataset = full_train
        if self.cfg.eval_jsonl:
            logger.info(f"Loading eval dataset: {self.cfg.eval_jsonl}")
            eval_dataset = load_dataset("json", data_files=self.cfg.eval_jsonl, split="train")
        elif self.cfg.eval_split_ratio and self.cfg.eval_split_ratio > 0.0:
            logger.info(f"Splitting train/eval with ratio={self.cfg.eval_split_ratio}")
            split = full_train.train_test_split(test_size=self.cfg.eval_split_ratio, seed=self.cfg.seed)
            train_dataset, eval_dataset = split["train"], split["test"]
        return train_dataset, eval_dataset

    def _build_training_args(self, eval_dataset=None):
        # Hardware safety: disable mixed precision if CUDA not available
        use_cuda = torch.cuda.is_available()
        bf16_ok = bool(self.cfg.bf16) and use_cuda and torch.cuda.is_bf16_supported()
        fp16_ok = bool(self.cfg.fp16) and use_cuda

        _sft_cfg_kwargs = dict(
            output_dir=self.cfg.output_dir,
            dataset_text_field=self.cfg.dataset_text_field,
            max_length=self.cfg.max_seq_length,
            per_device_train_batch_size=self.cfg.batch_size,
            gradient_accumulation_steps=self.cfg.grad_accum,
            learning_rate=self.cfg.lr,
            num_train_epochs=self.cfg.epochs,
            logging_steps=self.cfg.logging_steps,
            save_steps=self.cfg.save_steps,
            bf16=bf16_ok,
            fp16=fp16_ok,
            warmup_steps=self.cfg.warmup_steps,
            lr_scheduler_type=self.cfg.lr_scheduler_type,
            seed=self.cfg.seed,
            packing=bool(self.cfg.packing),
            save_total_limit=self.cfg.save_total_limit,
            load_best_model_at_end=(bool(self.cfg.load_best_at_end) if eval_dataset is not None else False),
            evaluation_strategy=("steps" if eval_dataset is not None else "no"),
            eval_steps=(self.cfg.eval_steps if eval_dataset is not None else None),
            save_strategy="steps",
            metric_for_best_model="loss",
            greater_is_better=False,
        )
        # Optional: enable gradient checkpointing to reduce activation memory if supported
        if hasattr(self.cfg, "gradient_checkpointing"):
            _sft_cfg_kwargs["gradient_checkpointing"] = bool(self.cfg.gradient_checkpointing)
        _allowed = set(inspect.signature(SFTConfig.__init__).parameters.keys())
        if ("evaluation_strategy" not in _allowed) or ("save_strategy" not in _allowed) or (eval_dataset is None):
            _sft_cfg_kwargs["load_best_model_at_end"] = False
            for k in ["evaluation_strategy", "eval_steps", "save_strategy", "metric_for_best_model", "greater_is_better"]:
                _sft_cfg_kwargs.pop(k, None)
        _filtered = {k: v for k, v in _sft_cfg_kwargs.items() if k in _allowed}
        return SFTConfig(**_filtered)

    def run(self):
        os.makedirs(self.cfg.output_dir, exist_ok=True)
        self._prepare_env()
        model, tokenizer = self._load_model_tokenizer()
        lora_cfg = self.build_lora_config(
            r=self.cfg.lora_r,
            alpha=self.cfg.lora_alpha,
            dropout=self.cfg.lora_dropout,
            target_modules=self.cfg.target_modules if self.cfg.target_modules else None,
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        train_dataset, eval_dataset = self._prepare_datasets()

        # Log step estimates
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        eff_batch = max(1, self.cfg.batch_size) * max(1, self.cfg.grad_accum) * max(1, world_size)
        steps_per_epoch = math.ceil(len(train_dataset) / eff_batch)
        total_steps = int(steps_per_epoch * self.cfg.epochs)
        logger.info(f"Samples: train={len(train_dataset)}{' | eval='+str(len(eval_dataset)) if eval_dataset is not None else ''}")
        logger.info(f"Devices: {world_size} | effective_batch={eff_batch} | steps_per_epoch={steps_per_epoch} | total_steps≈{total_steps}")

        training_args = self._build_training_args(eval_dataset)

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
            trainer = SFTTrainer(**trainer_init)

        class _EvalPrinter(TrainerCallback):
            def on_evaluate(self, args, state, control, metrics=None, **kwargs):
                if metrics is not None:
                    eval_loss = metrics.get("eval_loss")
                    logger.info(f"[Eval] step={state.global_step} eval_loss={eval_loss} metrics={metrics}")
                return control

        trainer.add_callback(_EvalPrinter())

        if self.cfg.early_stopping_patience and eval_dataset is not None:
            metric = getattr(training_args, "metric_for_best_model", None)
            eval_strategy = getattr(training_args, "evaluation_strategy", None)
            if metric is None or (eval_strategy in (None, "no")):
                logger.warning("Early stopping disabled: metric/evaluation_strategy not available.")
            else:
                trainer.add_callback(EarlyStoppingCallback(
                    early_stopping_patience=self.cfg.early_stopping_patience,
                    early_stopping_threshold=self.cfg.early_stopping_delta,
                ))

        if self.cfg.eval_before_train and 'eval_dataset' in trainer.__dict__ and trainer.eval_dataset is not None:
            logger.info("Pre-training evaluation enabled. Running evaluation before training…")
            try:
                pre_metrics = trainer.evaluate()
                logger.info(f"Pre-train eval metrics: {pre_metrics}")
                try:
                    import json as _json
                    pre_metrics_path = Path(self.cfg.output_dir) / "eval_metrics_before.json"
                    with open(pre_metrics_path, "w", encoding="utf-8") as f:
                        _json.dump(pre_metrics, f, indent=2, ensure_ascii=False)
                except Exception as e:
                    logger.warning(f"Failed to save pre-train eval metrics: {e}")
            except Exception as e:
                logger.warning(f"Pre-training evaluation failed: {e}")

        logger.info("Starting training...")
        trainer.train()

        logger.success("Training finished. Saving adapter...")
        trainer.model.save_pretrained(self.cfg.output_dir)
        tokenizer.save_pretrained(self.cfg.output_dir)
        logger.info(f"Saved to: {self.cfg.output_dir}")

        if 'eval_dataset' in trainer.__dict__ and trainer.eval_dataset is not None:
            logger.info("Running evaluation at end of training…")
            try:
                metrics = trainer.evaluate()
                try:
                    import json as _json
                    metrics_path = Path(self.cfg.output_dir) / "eval_metrics_final.json"
                    with open(metrics_path, "w", encoding="utf-8") as f:
                        _json.dump(metrics, f, indent=2, ensure_ascii=False)
                    logger.info(f"Saved eval metrics to: {metrics_path}")
                except Exception as e:
                    logger.warning(f"Failed to save eval metrics: {e}")
            except Exception as e:
                logger.warning(f"Evaluation after training failed: {e}")


__all__ = ["LoraFinetuner"]
# LoraFinetuner.from_hparams("/home/server1/AI2/OuteTTS/datas/wavs/lsy/lora_hparams.json").run()
