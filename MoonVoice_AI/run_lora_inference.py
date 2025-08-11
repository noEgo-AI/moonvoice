import os
import argparse
from pathlib import Path

import torch
from loguru import logger
from transformers import AutoModelForCausalLM
from peft import PeftModel

import outetts


def parse_args():
    p = argparse.ArgumentParser(
        description="Run inference with a local base model + LoRA adapter (OuteTTS v1.0, interface v3)"
    )
    p.add_argument("--model_path", required=True, help="Base model local dir or repo id")
    p.add_argument("--tokenizer_path", required=True, help="Tokenizer local dir or repo id")
    p.add_argument("--lora_dir", required=True, help="Path to the trained LoRA adapter (output_dir from lora.py)")
    p.add_argument("--text", required=True, help="Text to synthesize")
    p.add_argument("--out", default="output_lora.wav", help="Output WAV path")

    speaker = p.add_mutually_exclusive_group()
    speaker.add_argument("--speaker_json", help="Path to saved speaker JSON (preferred)")
    speaker.add_argument("--default_speaker", default="en-female-1-neutral", help="Default speaker name for v3")
    speaker.add_argument("--speaker_wav", help="Create speaker from a short WAV file (<=15s recommended)")

    p.add_argument("--device", default=None, help="Device, e.g., cuda or cpu (auto if omitted)")
    p.add_argument("--dtype", default=None, choices=["bf16", "fp16", "fp32"], help="Inference dtype")
    # Convenience flags compatible with training scripts
    p.add_argument("--bf16", action="store_true", help="Use bfloat16 (alias for --dtype bf16)")
    p.add_argument("--fp16", action="store_true", help="Use float16 (alias for --dtype fp16)")
    p.add_argument("--fp32", action="store_true", help="Use float32 (alias for --dtype fp32)")
    p.add_argument("--max_length", type=int, default=8192, help="Max generation length")
    p.add_argument("--temperature", type=float, default=0.4, help="Sampling temperature")
    p.add_argument("--force_local", action="store_true", help="Force offline mode and local files only")
    p.add_argument("--merge_lora", action="store_true", help="Merge LoRA weights into base model for inference")
    return p.parse_args()


def pick_dtype(name: str | None):
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    if name == "fp32":
        return torch.float32
    # Auto-pick based on hardware
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def main():
    args = parse_args()

    # Resolve dtype preference from flags if --dtype not provided
    if args.dtype is None:
        if args.bf16:
            args.dtype = "bf16"
        elif args.fp16:
            args.dtype = "fp16"
        elif args.fp32:
            args.dtype = "fp32"

    model_is_dir = Path(args.model_path).is_dir()
    tok_is_dir = Path(args.tokenizer_path).is_dir()
    lora_is_dir = Path(args.lora_dir).is_dir()

    if args.force_local or model_is_dir or tok_is_dir or lora_is_dir:
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("HF_HUB_OFFLINE", "1")

    # Validate local paths when they look like paths
    def looks_like_path(p: str) -> bool:
        return any(s in p for s in (os.sep, "/", "\\")) or p.startswith(".")

    if looks_like_path(args.model_path) and not model_is_dir:
        raise FileNotFoundError(f"Model path not found: {args.model_path}")
    if looks_like_path(args.tokenizer_path) and not tok_is_dir:
        raise FileNotFoundError(f"Tokenizer path not found: {args.tokenizer_path}")
    if not lora_is_dir:
        raise FileNotFoundError(f"LoRA adapter dir not found: {args.lora_dir}")

    # Build OuteTTS interface with HF backend (v3)
    dtype = pick_dtype(args.dtype)
    cfg = outetts.ModelConfig(
        model_path=args.model_path,
        tokenizer_path=args.tokenizer_path,
        interface_version=outetts.InterfaceVersion.V3,
        backend=outetts.Backend.HF,
        device=args.device,
        dtype=dtype,
    )
    interface = outetts.Interface(cfg)

    # Speaker handling
    speaker = None
    if args.speaker_json:
        speaker = interface.load_speaker(args.speaker_json)
    elif args.speaker_wav:
        speaker = interface.create_speaker(args.speaker_wav)
    else:
        speaker = interface.load_default_speaker(args.default_speaker)

    # Load base model and apply LoRA adapter
    local_only = True if (args.force_local or model_is_dir) else False
    logger.info(f"Loading base model from: {args.model_path}")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=local_only,
    )
    logger.info(f"Loading LoRA adapter from: {args.lora_dir}")
    model = PeftModel.from_pretrained(base_model, args.lora_dir)
    if args.merge_lora:
        logger.info("Merging LoRA weights into base model for inference")
        model = model.merge_and_unload()

    # Attach the adapted model to the interface
    # interface.model is outetts.models.hf_model.HFModel
    interface.model.model = model.to(interface.model.device)

    # Build generation config and run inference
    gen = outetts.GenerationConfig(
        text=args.text,
        speaker=speaker,
        generation_type=outetts.GenerationType.CHUNKED,
        sampler_config=outetts.SamplerConfig(temperature=args.temperature),
        max_length=args.max_length,
    )
    logger.info("Generating audio...")
    output = interface.generate(gen)
    output.save(args.out)
    logger.success(f"Done. Saved to: {args.out}")


if __name__ == "__main__":
    main()
