#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end demo for a single image:
- Load LLaVA-1.6 (HF)
- Ask: "What text is within this image."
- Capture projector outputs (image->LLM embeddings), visualize token-norm heatmap
- Do targeted embedding ablation (top-k dims by magnitude on this example) + random-k placebo
- Re-generate answers, compare deltas
- Try to collect LLM attentions; if unsupported, skip gracefully
- Save all outputs to ./fig/
"""

import os
import math
import json
import random
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    AutoConfig,
    AutoModelForCausalLM,
)

# --------------------------
# Config
# --------------------------
CKPT = os.environ.get("LLAVA_CKPT", "llava-hf/llava-v1.6-mistral-7b-hf")
IMAGE_PATH = "example.png"
QUESTION = "What text is within this image."
OUT_DIR = Path("./fig")
OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
TOPK = 32
MAX_NEW_TOKENS = 64

# --------------------------
# Utilities
# --------------------------
def load_model_and_processor():
    cfg = AutoConfig.from_pretrained(CKPT)
    # Try to enable attentions; if a given head doesn't support gen-attn, we'll catch later.
    cfg.output_attentions = True

    processor = AutoProcessor.from_pretrained(CKPT, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT,
        torch_dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    model.config.output_attentions = True
    model.eval()
    return model, processor, tokenizer

def find_projector_module(model: nn.Module):
    """
    Heuristic search for the LLaVA projector that maps vision features to LLM hidden size.
    Works across common LLaVA-1.6 HF variants.
    """
    candidates = []
    for name, m in model.named_modules():
        lname = name.lower()
        if ("mm_projector" in lname) or ("visual_projector" in lname) or ("mlp" in lname and "vision" in lname and "proj" in lname):
            # select linear/MLP modules
            if isinstance(m, (nn.Linear, nn.Sequential, nn.Module)):
                candidates.append((name, m))
    # Prefer mm_projector if present
    for n, m in candidates:
        if "mm_projector" in n:
            return n, m
    # Fallback: first candidate
    if candidates:
        return candidates[0]
    raise RuntimeError("Could not locate projector module; inspect model.named_modules().")

def pretty_print(s):
    print("=" * 80)
    print(s)
    print("=" * 80)

def answer_contains_text(ans: str):
    # Heuristic "hallucination" flag: anything not equal to NO-TEXT & contains alnum
    a = ans.strip().upper()
    if a == "NO-TEXT":
        return False
    return any(ch.isalnum() for ch in ans)

def square_grid(n_tokens: int):
    # LLaVA often uses ViT with square patch grid; infer side
    side = int(round(math.sqrt(n_tokens)))
    if side * side == n_tokens:
        return side, side
    # fallback: keep rectangular by factoring nearest
    for h in range(side, 1, -1):
        if n_tokens % h == 0:
            return h, n_tokens // h
    return side, side  # best effort

def save_heatmap(grid, title, out_path):
    plt.figure(figsize=(4.5, 4.2))
    plt.imshow(grid, interpolation="nearest")
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()

# --------------------------
# Hooks: capture & ablate projector outputs
# --------------------------
class ProjectorCapture:
    def __init__(self):
        self.tensor = None
        self.h_capture = None
        self.h_ablate = None
        self.target_module = None
        self.target_name = None

    def attach_capture(self, model):
        name, module = find_projector_module(model)
        self.target_module = module
        self.target_name = name

        def hook(_m, _inp, out):
            # Expect [B, N_img_tokens, d] or similar
            self.tensor = out.detach()
            return out

        self.h_capture = module.register_forward_hook(hook)

    def attach_ablation(self, dims_to_zero: torch.Tensor):
        assert self.target_module is not None, "Call attach_capture() first to locate the projector."
        # Remove old ablation if any
        if self.h_ablate is not None:
            self.h_ablate.remove()
            self.h_ablate = None

        def hook(_m, _inp, out):
            z = out
            z[..., dims_to_zero] = 0
            return z

        self.h_ablate = self.target_module.register_forward_hook(hook)

    def remove_all(self):
        if self.h_capture is not None:
            self.h_capture.remove()
            self.h_capture = None
        if self.h_ablate is not None:
            self.h_ablate.remove()
            self.h_ablate = None

# --------------------------
# Main
# --------------------------
def main():
    # 1) Load
    model, processor, tokenizer = load_model_and_processor()
    pretty_print(f"Loaded {CKPT} on {DEVICE}")

    # 2) Prepare inputs
    img = Image.open(IMAGE_PATH).convert("RGB")

    # Many LLaVA HF processors accept {"images": img, "text": prompt}
    prompt = QUESTION
    inputs = processor(images=img, text=prompt, return_tensors="pt").to(model.device)

    # 3) Register projector capture
    cap = ProjectorCapture()
    cap.attach_capture(model)
    pretty_print(f"Projector module hooked at: {cap.target_name}")

    # 4) Baseline generation
    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,  # may be unsupported; we'll catch later
        )

    # decode response
    seq = gen_out.sequences[0]
    # input_ids length:
    n_in = inputs["input_ids"].shape[1]
    baseline_answer = tokenizer.decode(seq[n_in:], skip_special_tokens=True).strip()
    pretty_print(f"Baseline answer:\n{baseline_answer}")

    # 5) Projector token map (norm heatmap)
    Z = cap.tensor  # [B, N, d]
    assert Z is not None, "Projector did not fire; check the hook or processor."
    Z0 = Z[0].float()  # [N, d]
    token_norms = torch.norm(Z0, dim=-1).cpu().numpy()  # [N]
    H, W = square_grid(len(token_norms))
    heat = token_norms.reshape(H, W)
    save_heatmap(heat, "Projector token L2-norms", OUT_DIR / "proj_token_norms.png")

    # 6) Targeted top-k dims (by magnitude on pooled embedding for this example)
    z_bar = Z0.mean(dim=0)  # [d]
    d = z_bar.numel()
    k = min(TOPK, d)
    topk_idx = torch.topk(z_bar.abs(), k).indices.to(model.device)
    np.save(OUT_DIR / "topk_dims.npy", topk_idx.detach().cpu().numpy())

    # 7) Re-run with targeted ablation
    cap.attach_ablation(topk_idx)
    with torch.no_grad():
        gen_out2 = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    seq2 = gen_out2.sequences[0]
    targeted_answer = tokenizer.decode(seq2[n_in:], skip_special_tokens=True).strip()
    cap.attach_ablation(torch.tensor([], dtype=torch.long, device=model.device))  # remove ablation by attaching empty set

    # 8) Random-k placebo
    rng = np.random.default_rng(0)
    rand_idx = torch.tensor(sorted(rng.choice(d, size=k, replace=False)), device=model.device)
    cap.attach_ablation(rand_idx)
    with torch.no_grad():
        gen_out3 = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,
        )
    seq3 = gen_out3.sequences[0]
    random_answer = tokenizer.decode(seq3[n_in:], skip_special_tokens=True).strip()
    cap.remove_all()

    # 9) Try to summarize (single-image demo)
    def flag(ans):
        return "YES" if answer_contains_text(ans) else "NO"

    summary = {
        "checkpoint": CKPT,
        "image": IMAGE_PATH,
        "question": QUESTION,
        "baseline_answer": baseline_answer,
        "baseline_hallucination_flag": flag(baseline_answer),
        "targeted_answer": targeted_answer,
        "targeted_hallucination_flag": flag(targeted_answer),
        "random_answer": random_answer,
        "random_hallucination_flag": flag(random_answer),
        "topk": int(k),
        "note": "Top-k dims chosen by |mean activation| on this example (single-image demo). "
                "For a proper study, fit a sparse probe over many symbol logos, then ablate those dims.",
    }
    with open(OUT_DIR / "single_example_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # 10) Attempt LLM attention: reliability varies by checkpoint/generation config
    attn_ok = False
    try:
        # gen_out was from baseline pass; some backends pack attentions differently.
        # We compute a simple aggregate: mean attention of last layer from generated tokens to all keys.
        if hasattr(gen_out, "attentions") and gen_out.attentions is not None:
            # Depending on transformers version, this may be a list of tuples (per-step); we handle common cases.
            att = gen_out.attentions
            if isinstance(att, (list, tuple)) and len(att) > 0:
                last = att[-1]
                # last usually: (num_layers, batch, num_heads, Q, K) or list per layer
                if isinstance(last, (list, tuple)):
                    last = last[-1]
                A = last  # tensor
                if isinstance(A, torch.Tensor) and A.dim() == 5:
                    # Average over batch & heads & query positions of new tokens
                    A_mean = A.mean(dim=(1, 2, 3)).squeeze(0).cpu().numpy()  # [K]
                    plt.figure(figsize=(6, 2.5))
                    plt.plot(A_mean)
                    plt.title("LLM attention (avg over heads/Q) to key positions")
                    plt.xlabel("Key index")
                    plt.ylabel("Mean attention")
                    plt.tight_layout()
                    plt.savefig(OUT_DIR / "llm_attention_keys.png", bbox_inches="tight")
                    plt.close()
                    attn_ok = True
    except Exception as e:
        print(f"[warn] Could not extract attentions reliably: {e}")

    # 11) Save a small text report
    report_lines = [
        f"Checkpoint: {CKPT}",
        f"Image: {IMAGE_PATH}",
        f"Question: {QUESTION}",
        "",
        f"[Baseline]  Answer: {baseline_answer}",
        f"[Targeted]  Answer: {targeted_answer}",
        f"[Random-k]  Answer: {random_answer}",
        "",
        f"Projector token norms heatmap: {OUT_DIR / 'proj_token_norms.png'}",
        f"Top-k dims saved: {OUT_DIR / 'topk_dims.npy'}",
        f"LLM attention figure: {'saved' if attn_ok else 'not available for this checkpoint'}",
        "",
        "Note: This single-image demo picks top-k dims by |mean activation|. "
        "For the paper experiment, learn top-k via a sparse logistic probe on a symbol-logo set, "
        "then re-run ablations with those dims."
    ]
    with open(OUT_DIR / "single_example_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    # 12) Console summary
    pretty_print("Summary")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\nSaved figures to:", OUT_DIR.resolve())
    print(" - proj_token_norms.png")
    print(" - llm_attention_keys.png (if available)")
    print("Saved text report:", (OUT_DIR / "single_example_report.txt").resolve())

if __name__ == "__main__":
    main()
