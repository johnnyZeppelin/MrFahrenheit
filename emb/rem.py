#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LLaVA demo (robust to LLaVA v1.6 vs LLaVA-Next):
- Loads model & processor with the correct class (not AutoModelForCausalLM)
- dtype=... (no torch_dtype)
- Baseline generation on example.png
- Capture projector outputs, save token-norm heatmap
- Targeted top-k (by |mean activation| on this example) ablation + random-k placebo
- Try to save an attention figure if supported
- Save everything to ./fig/
"""

import os, math, json, random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image

from transformers import AutoConfig, AutoProcessor, AutoTokenizer

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
CKPT = os.environ.get("LLAVA_CKPT",  # pick a compatible checkpoint:
                      # e.g. "llava-hf/llava-1.6-mistral-7b-hf"  (LLaVA v1.6)
                      # or  "llava-hf/llava-next-34b-hf"         (LLaVA-Next)
                      "llava-hf/llava-v1.6-mistral-7b-hf")
IMAGE_PATH = "example.png"
QUESTION = "What text is within this image."
OUT_DIR = Path("./fig"); OUT_DIR.mkdir(parents=True, exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if torch.cuda.is_available() else torch.float32
TOPK = 32
MAX_NEW_TOKENS = 64

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def pretty(msg): print("\n" + "="*80 + f"\n{msg}\n" + "="*80)

def answer_contains_text(ans: str) -> bool:
    s = ans.strip().upper()
    if s == "NO-TEXT": return False
    return any(ch.isalnum() for ch in ans)

def square_grid(n: int):
    side = int(round(math.sqrt(n)))
    if side*side == n: return side, side
    for h in range(side, 1, -1):
        if n % h == 0: return h, n//h
    return side, side

def save_heatmap(grid, title, path: Path):
    plt.figure(figsize=(4.6, 4.2))
    plt.imshow(grid, interpolation="nearest")
    plt.title(title)
    plt.axis("off"); plt.tight_layout()
    plt.savefig(path, bbox_inches="tight"); plt.close()

# -----------------------------------------------------------------------------
# Model loading with correct class
# -----------------------------------------------------------------------------
def load_llava_model_and_processor():
    cfg = AutoConfig.from_pretrained(CKPT)
    model_type = getattr(cfg, "model_type", "").lower()

    # pick class according to model_type
    llava_model_cls = None
    if model_type in {"llava"}:
        from transformers import LlavaForConditionalGeneration
        llava_model_cls = LlavaForConditionalGeneration
    elif model_type in {"llava_next", "llava-onevision", "llava_onevision", "llava-next"}:
        # HF uses LlavaNextForConditionalGeneration (and for OneVision too)
        from transformers import LlavaNextForConditionalGeneration
        llava_model_cls = LlavaNextForConditionalGeneration
    else:
        raise ValueError(f"Unsupported/unknown LLaVA model_type='{model_type}'. "
                         f"Please use a LLaVA v1.6 or LLaVA-Next checkpoint.")

    processor = AutoProcessor.from_pretrained(CKPT, use_fast=False)
    tokenizer = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
    # === added
    # cfg = AutoConfig.from_pretrained(CKPT)
    cfg.attn_implementation = "eager"   # <-- enable attention outputs
    cfg.output_attentions = True
    
    model = llava_model_cls.from_pretrained(
        CKPT,
        config=cfg,           # <-- pass the config
        dtype=DTYPE,
        device_map="auto",
        low_cpu_mem_usage=True,
    )
    # === end
    # model = llava_model_cls.from_pretrained(
    #     CKPT,
    #     dtype=DTYPE,              # <= use dtype (torch_dtype is deprecated)
    #     device_map="auto",
    #     low_cpu_mem_usage=True,
    # )
    # try to enable attentions if supported
    if hasattr(model.config, "output_attentions"):
        model.config.output_attentions = True
    model.eval()
    return model, processor, tokenizer, model_type

# -----------------------------------------------------------------------------
# Find projector & attach hooks
# -----------------------------------------------------------------------------
class ProjectorHooks:
    def __init__(self):
        self.capture_handle = None
        self.ablate_handle = None
        self.tensor = None
        self.module = None
        self.name = None

    def find_projector_module(self, model: nn.Module):
        # Look for common names across LLaVA variants
        cands = []
        for n, m in model.named_modules():
            ln = n.lower()
            if ("mm_projector" in ln) or ("visual_projector" in ln) or ("vision" in ln and "proj" in ln):
                cands.append((n, m))
        # prefer mm_projector if present
        for n, m in cands:
            if "mm_projector" in n: return n, m
        if cands: return cands[0]
        raise RuntimeError("Projector module not found. Inspect model.named_modules().")

    def attach_capture(self, model: nn.Module):
        self.name, self.module = self.find_projector_module(model)
        def hook(_m, _inp, out):
            self.tensor = out.detach()
            return out
        self.capture_handle = self.module.register_forward_hook(hook)

    def attach_ablation(self, dims_to_zero: torch.Tensor):
        # Remove old ablation if any
        self.remove_ablation()
        def hook(_m, _inp, out):
            z = out
            z[..., dims_to_zero] = 0
            return z
        self.ablate_handle = self.module.register_forward_hook(hook)

    def remove_ablation(self):
        if self.ablate_handle is not None:
            self.ablate_handle.remove()
            self.ablate_handle = None

    def remove_all(self):
        self.remove_ablation()
        if self.capture_handle is not None:
            self.capture_handle.remove()
            self.capture_handle = None

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    model, processor, tokenizer, model_type = load_llava_model_and_processor()
    pretty(f"Loaded {CKPT} ({model_type}) on {DEVICE}")

    # Inputs
    img = Image.open(IMAGE_PATH).convert("RGB")
    inputs = processor(images=img, text=QUESTION, return_tensors="pt").to(model.device)

    # Hook projector capture
    hooks = ProjectorHooks()
    hooks.attach_capture(model)
    pretty(f"Projector hooked at: {hooks.name}")

    # Baseline generation
    with torch.no_grad():
        out_base = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            return_dict_in_generate=True,
            output_attentions=True,   # may be ignored silently
        )
    n_in = inputs["input_ids"].shape[1]
    base_ans = tokenizer.decode(out_base.sequences[0][n_in:], skip_special_tokens=True).strip()
    pretty(f"[Baseline] {base_ans}")

    # Projector token-norm heatmap
    Z = hooks.tensor
    assert Z is not None, "Projector hook did not fire. Check projector path."
    Z0 = Z[0].float()                     # [N_img_tokens, d]
    norms = torch.norm(Z0, dim=-1).cpu().numpy()
    H, W = square_grid(len(norms))
    save_heatmap(norms.reshape(H, W), "Projector token L2-norms", OUT_DIR / "proj_token_norms.png")

    # Targeted top-k dims by |mean activation| on this example
    zbar = Z0.mean(dim=0)                 # [d]
    d = zbar.numel()
    k = min(TOPK, d)
    topk_idx = torch.topk(zbar.abs(), k).indices.to(model.device)
    np.save(OUT_DIR / "topk_dims.npy", topk_idx.detach().cpu().numpy())

    # Targeted ablation run
    hooks.attach_ablation(topk_idx)
    with torch.no_grad():
        out_targ = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            return_dict_in_generate=True, output_attentions=True
        )
    targ_ans = tokenizer.decode(out_targ.sequences[0][n_in:], skip_special_tokens=True).strip()
    hooks.remove_ablation()

    # Random-k placebo
    rng = np.random.default_rng(0)
    rand_idx = torch.tensor(sorted(rng.choice(d, size=k, replace=False)), device=model.device)
    hooks.attach_ablation(rand_idx)
    with torch.no_grad():
        out_rand = model.generate(
            **inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False,
            return_dict_in_generate=True, output_attentions=True
        )
    rand_ans = tokenizer.decode(out_rand.sequences[0][n_in:], skip_special_tokens=True).strip()
    hooks.remove_all()

    # Try an attention summary (robust to models that don't expose per-step attn)
    attn_saved = False
    try:
        att = getattr(out_base, "attentions", None)
        if att:
            last = att[-1]
            if isinstance(last, (list, tuple)): last = last[-1]
            if isinstance(last, torch.Tensor) and last.dim() == 5:
                # [layers, batch, heads, Q, K] -> mean over B,H,Q
                A = last.mean(dim=(1,2,3)).squeeze(0).cpu().numpy()  # [K]
                plt.figure(figsize=(6.2, 2.6))
                plt.plot(A); plt.title("LLM attention (avg over heads and queries)")
                plt.xlabel("Key index"); plt.ylabel("Mean attention")
                plt.tight_layout(); plt.savefig(OUT_DIR / "llm_attention_keys.png", bbox_inches="tight")
                plt.close()
                attn_saved = True
    except Exception as e:
        print(f"[warn] attention export failed: {e}")

    # Summarize
    def flag(ans): return "YES" if answer_contains_text(ans) else "NO"
    summary = {
        "checkpoint": CKPT,
        "image": IMAGE_PATH,
        "question": QUESTION,
        "baseline_answer": base_ans,
        "baseline_hallucination": flag(base_ans),
        "targeted_answer": targ_ans,
        "targeted_hallucination": flag(targ_ans),
        "random_answer": rand_ans,
        "random_hallucination": flag(rand_ans),
        "topk": int(k),
        "notes": "Single-image demo: top-k by |mean activation|. For the paper, learn top-k via a sparse probe over a symbol-logo set."
    }
    with open(OUT_DIR / "single_example_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    # Also write a tiny text report
    lines = [
        f"Checkpoint: {CKPT}",
        f"Image: {IMAGE_PATH}",
        f"Question: {QUESTION}",
        "",
        f"[Baseline]  {base_ans}",
        f"[Targeted]  {targ_ans}",
        f"[Random-k]  {rand_ans}",
        "",
        f"Projector token-norm heatmap saved: {OUT_DIR / 'proj_token_norms.png'}",
        f"Top-k dims saved: {OUT_DIR / 'topk_dims.npy'}",
        f"LLM attention figure: {'saved' if attn_saved else 'not available'}",
    ]
    with open(OUT_DIR / "single_example_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    pretty("Summary")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print("\nSaved to ./fig/:")
    for p in ["proj_token_norms.png", "topk_dims.npy", "llm_attention_keys.png", "single_example_summary.json", "single_example_report.txt"]:
        print(" -", OUT_DIR / p)

if __name__ == "__main__":
    main()
