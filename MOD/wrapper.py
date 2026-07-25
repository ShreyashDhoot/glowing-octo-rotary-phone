#!/usr/bin/env python
"""
mod_sweep.py
============

Standalone, minimal-dependency wrapper around the "Multi-Objective Decoding"
(MOD) repo's weighted logit-fusion decoding (reverse_kl f_type, i.e.
`scripts/eval/mod.py` in the original repo), rewritten so you don't need
accelerate/trl/tyro/wandb/fastchat/safe-rlhf/flash-attn/deepspeed etc.
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from typing import List, Optional

import torch
from tqdm import tqdm

# -----------------------------------------------------------------------
# Default sweep: (harmless_weight, helpful_weight) pairs, as requested.
# -----------------------------------------------------------------------
DEFAULT_SWEEP = [
    (1.0, 0.0),
    (0.75, 0.25),
    (0.5, 0.5),
    (0.25, 0.75),
    (0.0, 1.0),
]


# -----------------------------------------------------------------------
# Prompt extraction from Anthropic/hh-rlhf (harmless-base subset)
# -----------------------------------------------------------------------
def extract_prompt(text: str) -> str:
    """Extracts everything up to and including the last 'Assistant:'"""
    parts = text.rsplit("\n\nAssistant:", 1)
    if len(parts) == 2:
        return parts[0] + "\n\nAssistant:"
    return text


def load_harmless_base_prompts(num_prompts: int, split: str = "test", 
                               seed: int = 42, revision: Optional[str] = None) -> List[str]:
    from datasets import load_dataset

    print(f"[mod_sweep] Loading Anthropic/hh-rlhf (data_dir=harmless-base, split={split}, seed={seed}) ...",
          file=sys.stderr)
    dataset = load_dataset(
        "Anthropic/hh-rlhf",
        data_dir="harmless-base",
        split=split,
        revision=revision,
    )
    
    # Mirror the benchmark script: shuffle with seed, then select the first N
    dataset = dataset.shuffle(seed=seed).select(
        range(min(num_prompts, len(dataset)))
    )

    prompts = [extract_prompt(row["chosen"]) for row in dataset]
    return prompts


def load_prompts_from_file(path: str, num_prompts: int) -> List[str]:
    prompts = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            prompts.append(obj["prompt"])
            if len(prompts) >= num_prompts:
                break
    return prompts


# -----------------------------------------------------------------------
# Model loading
# -----------------------------------------------------------------------
@dataclass
class LoadedModel:
    model: "PeftModel"
    tokenizer: "AutoTokenizer"
    device: torch.device


def load_model(args) -> LoadedModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    quant_kwargs = {}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        quant_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch_dtype,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )

    print(f"[mod_sweep] Loading base model {args.base_model} ...", file=sys.stderr)
    device_map = "auto" if args.device == "cuda" else None
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch_dtype,
        device_map=device_map,
        **quant_kwargs,
    )
    if device_map is None:
        base_model = base_model.to(args.device)

    base_model.config.use_cache = True
    if base_model.config.pad_token_id is None:
        base_model.config.pad_token_id = base_model.config.eos_token_id

    print(f"[mod_sweep] Loading adapter '{args.safer_adapter}' as 'safer' ...", file=sys.stderr)
    model = PeftModel.from_pretrained(base_model, args.safer_adapter, adapter_name="safer")
    print(f"[mod_sweep] Loading adapter '{args.better_adapter}' as 'better' ...", file=sys.stderr)
    model.load_adapter(args.better_adapter, adapter_name="better")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    device = next(model.parameters()).device
    return LoadedModel(model=model, tokenizer=tokenizer, device=device)


# -----------------------------------------------------------------------
# Weighted logit-fusion greedy decoding (reverse_kl, single sequence)
# -----------------------------------------------------------------------
@torch.no_grad()
def fused_generate(
    lm: LoadedModel,
    prompt_text: str,
    w_harmless: float,
    w_helpful: float,
    max_new_tokens: int = 128,
) -> str:
    model, tokenizer, device = lm.model, lm.tokenizer, lm.device

    enc = tokenizer(prompt_text, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = input_ids.shape[1]

    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id

    pkv = {"safer": None, "better": None}
    cur_input_ids = {"safer": input_ids, "better": input_ids}
    cur_attn_mask = {"safer": attention_mask, "better": attention_mask}

    generated_ids = input_ids
    finished = False

    for _ in range(max_new_tokens):
        step_logp = {}
        for name in ("safer", "better"):
            model.set_adapter(name)
            out = model(
                input_ids=cur_input_ids[name],
                attention_mask=cur_attn_mask[name],
                past_key_values=pkv[name],
                use_cache=True,
            )
            pkv[name] = out.past_key_values
            logits = out.logits[:, -1, :]
            step_logp[name] = torch.log_softmax(logits.float(), dim=-1)

        combined = w_harmless * step_logp["safer"] + w_helpful * step_logp["better"]
        next_token = torch.argmax(combined, dim=-1)  # (1,)

        if eos_token_id is not None and next_token.item() == eos_token_id:
            finished = True

        generated_ids = torch.cat([generated_ids, next_token.unsqueeze(0)], dim=-1)

        next_tok_2d = next_token.unsqueeze(0)  # (1,1)
        for name in ("safer", "better"):
            cur_input_ids[name] = next_tok_2d
            cur_attn_mask[name] = torch.cat(
                [cur_attn_mask[name], torch.ones_like(next_tok_2d)], dim=-1
            )

        if finished:
            break

    response_ids = generated_ids[0, prompt_len:]
    response_text = tokenizer.decode(response_ids, skip_special_tokens=True)
    return response_text.strip()


# -----------------------------------------------------------------------
# Main sweep driver
# -----------------------------------------------------------------------
def parse_weights_arg(weights_str: Optional[str]):
    if not weights_str:
        return DEFAULT_SWEEP
    pairs = []
    for chunk in weights_str.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        a, b = chunk.split(",")
        pairs.append((float(a), float(b)))
    return pairs


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base_model", type=str, default="PKU-Alignment/alpaca-7b-reproduced",
                         help="Base SFT model the LoRA adapters were trained on top of.")
    parser.add_argument("--safer_adapter", type=str, default="./DPO-safer",
                         help="Path to the harmlessness-tuned LoRA adapter dir.")
    parser.add_argument("--better_adapter", type=str, default="./DPO-better",
                         help="Path to the helpfulness-tuned LoRA adapter dir.")

    parser.add_argument("--dataset_name", type=str, default="Anthropic/hh-rlhf")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num_prompts", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42, 
                         help="Seed for dataset shuffling (matches benchmark script).")
    parser.add_argument("--prompts_file", type=str, default=None,
                         help="Optional local .jsonl of {\"prompt\": ...} lines, "
                              "bypasses the Hub download entirely.")
    parser.add_argument("--dump_prompts_only", action="store_true",
                         help="Just fetch/cache prompts to --prompts_file and exit "
                              "(no model loaded, no generation).")

    parser.add_argument("--weights", type=str, default=None,
                         help="Override sweep, e.g. '1.0,0.0;0.5,0.5;0.0,1.0' "
                              "as 'harmless,helpful' pairs. Default is the 5 "
                              "pairs: (1,0) (0.75,0.25) (0.5,0.5) (0.25,0.75) (0,1).")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--dtype", type=str, default="bfloat16",
                         choices=["float16", "bfloat16", "float32"])
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--load_in_4bit", action="store_true",
                         help="Load base model in 4-bit (needs bitsandbytes). "
                              "Big help if you're low on GPU memory.")

    parser.add_argument("--output_dir", type=str, default="./sweep_outputs")
    args = parser.parse_args()

    # ---- prompts ----
    if args.prompts_file and os.path.exists(args.prompts_file) and not args.dump_prompts_only:
        print(f"[mod_sweep] Loading prompts from local file {args.prompts_file}", file=sys.stderr)
        prompts = load_prompts_from_file(args.prompts_file, args.num_prompts)
    else:
        prompts = load_harmless_base_prompts(args.num_prompts, split=args.split, seed=args.seed)
        if args.prompts_file:
            os.makedirs(os.path.dirname(os.path.abspath(args.prompts_file)) or ".", exist_ok=True)
            with open(args.prompts_file, "w") as f:
                for p in prompts:
                    f.write(json.dumps({"prompt": p}) + "\n")
            print(f"[mod_sweep] Cached {len(prompts)} prompts to {args.prompts_file}", file=sys.stderr)

    print(f"[mod_sweep] Using {len(prompts)} prompts.", file=sys.stderr)

    if args.dump_prompts_only:
        print("[mod_sweep] --dump_prompts_only set, exiting before model load.", file=sys.stderr)
        return

    weight_pairs = parse_weights_arg(args.weights)
    print(f"[mod_sweep] Sweep (harmless_weight, helpful_weight): {weight_pairs}", file=sys.stderr)

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- model ----
    lm = load_model(args)

    for w_harmless, w_helpful in weight_pairs:
        tag = f"h{w_harmless:.2f}_u{w_helpful:.2f}"
        out_path = os.path.join(args.output_dir, f"mod_sweep_{tag}.json")
        print(f"[mod_sweep] === Config harmless={w_harmless} helpful={w_helpful} -> {out_path} ===",
              file=sys.stderr)

        results = []
        for prompt_raw in tqdm(prompts, desc=tag):
            # Directly pass the multi-turn prompt without an Alpaca wrapper
            response = fused_generate(
                lm, prompt_raw, w_harmless, w_helpful,
                max_new_tokens=args.max_new_tokens,
            )
            results.append({
                "prompt": prompt_raw,
                "response": response,
                "harmless_weight": w_harmless,
                "helpful_weight": w_helpful,
            })

        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[mod_sweep] Wrote {len(results)} examples to {out_path}", file=sys.stderr)

    print("[mod_sweep] Done.", file=sys.stderr)


if __name__ == "__main__":
    main()