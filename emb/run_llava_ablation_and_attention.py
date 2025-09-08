#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
End-to-end demo for a single image with LLaVA-1.6 (Mistral-7B) on:
  - baseline generation
  - projector-embedding hooks
  - attention-to-image-token diagnostics
  - targeted top-k embedding ablation (single-image fallback)
  - random-k placebo ablation
Outputs figures to ./fig/

Requirements:
  pip install --upgrade "transformers>=4.41" accelerate sentencepiece torch torchvision torchaudio
"""

import os
import math
import random
import re
from pathlib import Path
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoConfig,
    LlavaForConditionalGeneration,
)

# -----------------------
# Config
# -----------------------
MODEL_ID = "llava-hf/llava-v1.6-mistral-7b-hf"
IMAGE_PATH = "example.png"
QUESTION = "What text is within this image."
FIG_DIR = Path("./fig")
FIG_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32
MAX_NEW_TOKENS = 64
TOPK = 32             # targeted ablation K
SEED = 1234
random.seed(SEED)
torch.manual_seed(SEED)

# ----------------------
# Added
# ----------------------
# --- (Optional but helpful) force PyTorch to avoid flash/efficient SDP kernels ---
try:
    from torch.backends.cuda import sdp_kernel
    sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
except Exception:
    pass  # older torch, ignore
# 1) Build config FIRST, set eager + attentions here
config = AutoConfig.from_pretrained(MODEL_ID)
config.attn_implementation = "eager"   # <-- critical
config.output_attentions = True        # <-- boolean True (NOT "eager")




# -----------------------
# Utilities
# -----------------------
def hallucinated(answer: str) -> bool:
    """
    A simple heuristic for the demo:
      - If it's exactly 'NO-TEXT' (case-insensitive), treat as non-hallucination for pure symbol.
      - Otherwise, if it contains alphanumeric, treat as hallucination.
    """
    s = answer.strip()
    if s.upper() == "NO-TEXT":
        return False
    return bool(re.search(r"[A-Za-z0-9]", s))

def save_txt_lines(path: Path, lines: List[str]):
    path.write_text("\n".join(lines), encoding="utf-8")

def ensure_tensor(x):
    return x if torch.is_tensor(x) else torch.tensor(x)

# -----------------------
# Load model & processor
# -----------------------
print(f"[INFO] Loading model: {MODEL_ID} on {DEVICE} ...")
config = AutoConfig.from_pretrained(MODEL_ID)
# added
config.attn_implementation = "eager"   # FIX: allow attention outputs
# ended
config.output_attentions = True  # request attentions
model = LlavaForConditionalGeneration.from_pretrained(
    MODEL_ID,
    config=config,                # <-- pass the config in
    torch_dtype=DTYPE,
    low_cpu_mem_usage=True,
    device_map="auto"
)
# Make sure model returns attentions
model.config.output_attentions = True
model.eval()

processor = AutoProcessor.from_pretrained(MODEL_ID)

# try
print("attn_impl:", getattr(model.config, "attn_implementation", None))
print("output_attentions:", model.config.output_attentions)
#

# -----------------------
# Projector hook (capture & optionally modify)
# -----------------------
proj_capture = {"z": None}
proj_module: Optional[nn.Module] = None

def find_projector_module(m: nn.Module) -> nn.Module:
    """
    Try common names in HF LLaVA-1.6 for the projector that maps vision features into LLM space.
    """
    candidate = None
    for name, module in m.named_modules():
        # Common names seen in LLaVA HF checkpoints:
        if "multi_modal_projector" in name and isinstance(module, nn.Module):
            candidate = module
    if candidate is None:
        # Fall back to any module with 'projector' in its name
        for name, module in m.named_modules():
            if "projector" in name:
                candidate = module
    if candidate is None:
        raise RuntimeError("Could not find projector module. Inspect model.named_modules().")
    print(f"[INFO] Using projector module: {candidate.__class__.__name__}")
    return candidate

def projector_capture_hook(_m, _inp, out):
    """
    Store the projector output: expected shape [B, N_img_tokens, d]
    """
    proj_capture["z"] = out.detach()
    return out

def projector_zero_dims_hook(dims_to_zero: torch.Tensor):
    dims_to_zero = dims_to_zero.to(next(model.parameters()).device)
    def _hook(_m, _inp, out):
        out = out.clone()
        out[..., dims_to_zero] = 0
        return out
    return _hook

proj_module = find_projector_module(model)
# Register a passive capture hook by default
handle_capture = proj_module.register_forward_hook(projector_capture_hook)

# -----------------------
# Encode image+prompt
# -----------------------
def resolve_image_token_string_and_id():
    """
    Returns (image_token_str, image_token_id) in a way that matches the model.
    """
    # 1) Prefer the model’s configured ID
    image_token_id = getattr(model.config, "image_token_id", None)

    # 2) Try to get the processor’s string form
    image_token_str = getattr(processor, "image_token", None)

    # 3) If the processor didn’t expose it, try common defaults (LLaVA HF usually uses <image>)
    if image_token_str is None:
        for cand in ("<image>", "<image_token>", "<image_placeholder>"):
            tid = processor.tokenizer.convert_tokens_to_ids(cand)
            if tid is not None and tid != processor.tokenizer.unk_token_id:
                image_token_str = cand
                break

    # 4) If we still don’t have the string but we DO have the id, try to map id->string
    if image_token_str is None and image_token_id is not None and image_token_id >= 0:
        # brute force find any token whose id is image_token_id
        vocab = processor.tokenizer.get_vocab()
        inv_vocab = {v: k for k, v in vocab.items()}
        image_token_str = inv_vocab.get(image_token_id, None)

    # 5) Final sanity: if we have a string, make sure tokenizer maps it to the model’s id
    if image_token_str is not None and image_token_id is not None and image_token_id >= 0:
        tid = processor.tokenizer.convert_tokens_to_ids(image_token_str)
        if tid != image_token_id:
            # tokenizer doesn’t know this special token with the correct id — register it explicitly
            # NOTE: this is rare for official checkpoints; but if needed:
            processor.tokenizer.add_special_tokens({"additional_special_tokens": [image_token_str]})
            model.resize_token_embeddings(len(processor.tokenizer))
            # re-lookup
            tid = processor.tokenizer.convert_tokens_to_ids(image_token_str)
            # if still mismatched, we’ll just trust tokenizer’s id for counting
            image_token_id = tid

    return image_token_str, image_token_id
    
# def build_inputs(image_path: str, question: str):
#     img = Image.open(image_path).convert("RGB")
#     # LLaVA HF expects a special image token in the prompt
#     # Processor will insert it if you use chat template:
#     prompt = f"USER: <image>\n{question}\nASSISTANT:"
#     # added
#     enc = processor(text=prompt, images=img, return_tensors="pt")
#     # Move to device without changing dtypes blindly
#     for k, v in enc.items():
#         if torch.is_tensor(v):
#             enc[k] = v.to(model.device)
#     # (A) CLIP vision tower does NOT accept image_sizes -> drop it if present
#     if "image_sizes" in enc:
#         enc.pop("image_sizes")
#     # (B) Ensure pixel_values are float (vision tower typically expects float32/16)
#     # Avoid casting input_ids (must stay torch.long)
#     if "pixel_values" in enc:
#         # Use float32 for maximum compatibility; float16 usually also works on A100
#         enc["pixel_values"] = enc["pixel_values"].to(model.device, dtype=torch.float32)
#     return enc, prompt
#     # ended
#     # inputs = processor(text=prompt, images=img, return_tensors="pt").to(model.device, dtype=DTYPE)
#     # return inputs, prompt
def build_inputs(image_path: str, question: str):
    img = Image.open(image_path).convert("RGB")
    image_token_str, image_token_id = resolve_image_token_string_and_id()
    # Use chat template so the processor inserts the right markers around the image placeholder
    chat = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": question},
            ],
        }
    ]
    prompt = processor.apply_chat_template(chat, add_generation_prompt=True, tokenize=False)

    enc = processor(text=prompt, images=[img], return_tensors="pt", padding=True)

    # ---- sanitize for CLIP ----
    # keep dtypes: only pixel_values -> float, input_ids must remain long
    if "image_sizes" in enc:
        enc.pop("image_sizes")
    if "pixel_values" in enc:
        pv = enc["pixel_values"].to(model.device)
        # ensure [B, C, H, W]
        if pv.ndim == 5:
            pv = pv[:, 0, ...]
        enc["pixel_values"] = pv.to(dtype=torch.float32)
    # move other tensors to device
    for k, v in list(enc.items()):
        if torch.is_tensor(v) and k != "pixel_values":
            enc[k] = v.to(model.device)

    return enc, prompt, img, image_token_id


def get_image_token_id():
    # Robustly find the image token id for this checkpoint
    tok = processor.tokenizer
    # Common cases:
    for cand in ("<image>", "<image_token>", "<image_placeholder>"):
        tid = tok.convert_tokens_to_ids(cand)
        if tid is not None and tid != tok.unk_token_id:
            return tid
    # Fallback: try special_tokens_map
    for k, v in getattr(tok, "special_tokens_map", {}).items():
        if isinstance(v, str) and "image" in v:
            tid = tok.convert_tokens_to_ids(v)
            if tid is not None and tid != tok.unk_token_id:
                return tid
    return None

# def debug_image_token_count(inputs):
#     tid = get_image_token_id()
#     if tid is None:
#         print("[WARN] Could not resolve image token id; skipping token count sanity check.")
#         return
#     count = int((inputs["input_ids"] == tid).sum().item())
#     print(f"[DEBUG] #image placeholder tokens in input_ids: {count}")
def debug_image_token_count(inputs, image_token_id):
    if image_token_id is None or image_token_id < 0:
        print("[WARN] image_token_id not available; skipping count.")
        return
    c = int((inputs["input_ids"] == image_token_id).sum().item())
    print(f"[DEBUG] #image placeholder tokens (id={image_token_id}) in input_ids: {c}")
    

# -----------------------
# Generate with attentions
# -----------------------
def generate_with_attn(inputs) -> Tuple[str, list]:
    """
    Run generation and return the decoded answer and attentions (decoder self-attn).
    """
    out = model.generate(
        **inputs,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        output_attentions=True,
        return_dict_in_generate=True
    )
    # out.sequences: [1, input_len + gen_len]
    # Note: out.attentions may be list (per step). We'll reduce to last-step attentions for plotting.
    sequences = out.sequences
    # Decode only the newly generated part for readability
    input_len = inputs["input_ids"].shape[1]
    gen_ids = sequences[0, input_len:]
    answer = processor.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    # Collect last-step attentions if available; otherwise take the last item
    attentions = out.attentions
    if isinstance(attentions, (list, tuple)) and len(attentions) > 0:
        last_attn = attentions[-1]  # tuple of layers? or tensor depending on version
    else:
        last_attn = None
    return answer, last_attn

# -----------------------
# Attention-to-image mass
# -----------------------
def attention_mass_to_image(last_attn, n_img_tokens: int, total_seq_len: int):
    """
    Compute per-layer average attention mass from (last-step) queries to the first n_img_tokens keys.
    last_attn is expected [num_layers, batch, num_heads, q_len, k_len] or similar.
    We sum over k in [0, n_img_tokens), average over heads and (optionally) over queries.
    """
    if last_attn is None:
        return None, None

    # Normalize shapes across possible return formats
    if isinstance(last_attn, (list, tuple)):
        # Some versions return a list of per-layer tensors
        # shape [B, H, Q, K] per layer
        mats = []
        for lay in last_attn:
            A = lay  # [B, H, Q, K]
            if isinstance(A, (list, tuple)):  # very old variants
                A = A[0]
            mats.append(A)
        attn = torch.stack(mats, dim=0)  # [L, B, H, Q, K]
    else:
        attn = last_attn  # assume already [L, B, H, Q, K]

    attn = attn.float().detach().cpu()
    L, B, H, Q, K = attn.shape
    # Sanity clamp
    n_img = min(n_img_tokens, K)
    # Average over Q (last few generated tokens could also be selected; for demo average all Q)
    A_meanQ = attn.mean(dim=3)  # [L, B, H, K]
    # Sum mass over image-key region
    img_mass = A_meanQ[..., :n_img].sum(dim=-1)  # [L, B, H]
    # Average over heads and batch
    per_layer_mass = img_mass.mean(dim=(1,2))  # [L]
    # For a quick 1D token view, take the last layer's per-token mass:
    last_layer = attn[-1, 0]  # [H, Q, K]
    token_mass = last_layer.mean(dim=0).mean(dim=0)  # [K]
    img_token_curve = token_mass[:n_img]  # [n_img]
    return per_layer_mass.numpy(), img_token_curve.numpy()

# -----------------------
# Visualization helpers
# -----------------------
def plot_layer_mass(layer_mass, path: Path):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(layer_mass)+1), layer_mass, marker="o")
    plt.xlabel("Decoder layer")
    plt.ylabel("Attention mass to image tokens")
    plt.title("Per-layer attention mass → image tokens")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_imgtoken_curve(curve, path: Path):
    plt.figure(figsize=(8, 3))
    plt.bar(range(1, len(curve)+1), curve)
    plt.xlabel("Image token index")
    plt.ylabel("Attention mass (last layer)")
    plt.title("Image-token attention (last layer, avg heads)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

def plot_ablation_delta(baseline_hall: int, targeted_hall: int, random_hall: int, path: Path):
    """
    Simple bar showing change in hallucination indicator (1/0) for this single example.
    ΔHall is baseline - condition (so positive bar = reduction).
    """
    base = baseline_hall
    deltas = [
        ("Targeted k=32", base - targeted_hall),
        ("Random k=32", base - random_hall),
    ]
    labels = [x[0] for x in deltas]
    vals = [x[1] for x in deltas]
    plt.figure(figsize=(6, 3.2))
    plt.bar(labels, vals)
    plt.ylabel("Δ Hallucination (baseline − condition)")
    plt.title("Ablation delta (single example)")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight")
    plt.close()

# -----------------------
# Targeted top-k (single-image fallback)
# -----------------------
def select_topk_dims_from_activations(z_tokens: torch.Tensor, k: int) -> torch.Tensor:
    """
    Single-image fallback if you don't have a trained probe:
      - z_tokens: [N_img_tokens, d]
      - score each dim by mean absolute activation across tokens
      - return top-k indices
    """
    with torch.no_grad():
        score = z_tokens.abs().mean(dim=0)  # [d]
        topk = torch.topk(score, k=min(k, score.numel()), largest=True).indices
    return topk

# -----------------------
# sanitizer
# -----------------------
def sanitize_inputs(enc: dict):
    enc = enc.copy()
    # 1) LLaVA → CLIP mismatch: drop keys CLIP doesn't accept
    if "image_sizes" in enc:
        enc.pop("image_sizes")

    # 2) Ensure pixel_values is 4-D [B, C, H, W] and float
    if "pixel_values" in enc:
        pv = enc["pixel_values"]
        # Move to device first
        pv = pv.to(model.device)
        # If 5-D [B, N_img, C, H, W] and N_img==1 → squeeze
        if pv.ndim == 5 and pv.shape[1] == 1:
            pv = pv[:, 0, ...]  # → [B, C, H, W]
        # If still 5-D with N_img>1, pick the first image for this demo
        if pv.ndim == 5 and pv.shape[1] > 1:
            pv = pv[:, 0, ...]
        # Cast only pixel_values to float32/16
        enc["pixel_values"] = pv.to(dtype=torch.float32)

    # 3) Move other tensors to device without changing dtype
    for k, v in list(enc.items()):
        if torch.is_tensor(v) and k != "pixel_values":
            enc[k] = v.to(model.device)

    return enc

# --- Force image-placeholder count to equal the vision feature tokens ---
@torch.no_grad()
def expected_img_token_count(pixel_values: torch.Tensor) -> int:
    """
    Runs the vision tower to infer how many patch tokens are produced.
    For CLIP, last_hidden_state has [CLS] + patches, so we subtract 1.
    """
    vt = model.model.vision_tower  # LlavaForConditionalGeneration -> LlavaLlamaModel -> vision_tower
    outs = vt(pixel_values=pixel_values, output_hidden_states=False)
    seq_len = outs.last_hidden_state.shape[1]  # [B, seq, C]
    # Heuristic for CLIP: first token is CLS
    return max(0, seq_len - 1)

def fix_image_placeholders(inputs: dict, image_token_id: int) -> dict:
    """
    Make the number of image placeholders in input_ids exactly equal to the
    number of patch tokens produced by the vision tower on pixel_values.
    If too many, compress to a single contiguous block of the correct length.
    If too few, pad the block to the correct length.
    """
    enc = {k: v for k, v in inputs.items()}
    assert "pixel_values" in enc and "input_ids" in enc, "inputs missing keys"
    exp_n = expected_img_token_count(enc["pixel_values"])
    ids = enc["input_ids"][0].tolist()

    # Find all positions with image_token_id
    pos = [i for i, t in enumerate(ids) if t == image_token_id]
    if len(pos) == exp_n:
        return enc  # already perfect

    # If none found, insert one block after the first BOS
    if len(pos) == 0:
        # insert right after the first token
        insert_at = 1 if len(ids) > 1 else 0
        new_ids = ids[:insert_at] + [image_token_id] * exp_n + ids[insert_at:]
        enc["input_ids"] = torch.tensor([new_ids], device=enc["input_ids"].device, dtype=enc["input_ids"].dtype)
        if "attention_mask" in enc:
            am = enc["attention_mask"][0].tolist()
            new_am = am[:insert_at] + [1] * exp_n + am[insert_at:]
            enc["attention_mask"] = torch.tensor([new_am], device=enc["attention_mask"].device, dtype=enc["attention_mask"].dtype)
        return enc

    # If we have many scattered, compress them into a single block of length exp_n
    first = pos[0]
    last = pos[-1]

    # Build new sequence: everything before first, then one block of exp_n image tokens,
    # then everything after last.
    new_ids = ids[:first] + [image_token_id] * exp_n + ids[last+1:]
    enc["input_ids"] = torch.tensor([new_ids], device=enc["input_ids"].device, dtype=enc["input_ids"].dtype)

    if "attention_mask" in enc:
        am = enc["attention_mask"][0].tolist()
        new_am = am[:first] + [1] * exp_n + am[last+1:]
        enc["attention_mask"] = torch.tensor([new_am], device=enc["attention_mask"].device, dtype=enc["attention_mask"].dtype)

    return enc



# -----------------------
# Run baseline
# -----------------------
print("[INFO] Running baseline ...")
# inputs, prompt = build_inputs(IMAGE_PATH, QUESTION)
# changed
inputs, prompt, img, image_token_id = build_inputs(IMAGE_PATH, QUESTION)

# Sanity prints (optional)
debug_image_token_count(inputs, image_token_id)
pv = inputs.get("pixel_values", None)
if pv is not None:
    print("pixel_values shape:", tuple(pv.shape), pv.dtype, pv.device)
print("input_ids shape/dtype:", tuple(inputs["input_ids"].shape), inputs["input_ids"].dtype)

# 🔧 FIX: normalize placeholder count to match vision features
inputs = fix_image_placeholders(inputs, image_token_id)

# Verify count again — should now be the expected number (likely 576)
debug_image_token_count(inputs, image_token_id)
# debug_image_token_count(inputs, image_token_id)
# # (Optional) print shapes
# pv = inputs.get("pixel_values", None)
# if pv is not None:
#     print("pixel_values shape:", tuple(pv.shape), pv.dtype, pv.device)
# print("input_ids shape/dtype:", tuple(inputs["input_ids"].shape), inputs["input_ids"].dtype)
# ended
# added
inputs = sanitize_inputs(inputs)
pv = inputs.get("pixel_values", None)
if pv is not None:
    print("pixel_values shape:", tuple(pv.shape), pv.dtype, pv.device)
print("input_ids shape/dtype:", tuple(inputs["input_ids"].shape), inputs["input_ids"].dtype)
# ended
proj_capture["z"] = None
baseline_answer, last_attn = generate_with_attn(inputs)
print(f"[BASELINE ANSWER]\n{baseline_answer}\n")

# infer number of image tokens from captured projector output
assert proj_capture["z"] is not None, "Projector output not captured."
z = proj_capture["z"]  # [B, N, d]
B, N_img_tokens, d = z.shape
print(f"[INFO] Projector tokens: N_img={N_img_tokens}, d={d}")

# attention diagnostics
if last_attn is not None:
    layer_mass, img_curve = attention_mass_to_image(last_attn, N_img_tokens, total_seq_len=inputs["input_ids"].shape[1])
    if layer_mass is not None:
        plot_layer_mass(layer_mass, FIG_DIR / "attention_image_token_mass.pdf")
    if img_curve is not None:
        plot_imgtoken_curve(img_curve, FIG_DIR / "image_token_attention_bar.pdf")
else:
    print("[WARN] Attentions were not returned by generate(); skipping attention plots.")

baseline_hall = 1 if hallucinated(baseline_answer) else 0

# -----------------------
# Targeted ablation (top-k dims from activations)
# -----------------------
print("[INFO] Running targeted ablation (top-k from activations) ...")
topk_idx = select_topk_dims_from_activations(z[0], TOPK)  # [k]
# swap capture hook for a modifying hook
handle_capture.remove()
handle_targeted = proj_module.register_forward_hook(projector_zero_dims_hook(topk_idx))
# run
proj_capture["z"] = None
tgt_answer, _ = generate_with_attn(inputs)
print(f"[TARGETED ANSWER]\n{tgt_answer}\n")
targeted_hall = 1 if hallucinated(tgt_answer) else 0
# clean up
handle_targeted.remove()
handle_capture = proj_module.register_forward_hook(projector_capture_hook)

# -----------------------
# Random placebo ablation
# -----------------------
print("[INFO] Running random-k placebo ...")
rand_idx = torch.tensor(sorted(random.sample(range(d), min(TOPK, d))), device=model.device)
handle_random = proj_module.register_forward_hook(projector_zero_dims_hook(rand_idx))
proj_capture["z"] = None
rnd_answer, _ = generate_with_attn(inputs)
print(f"[RANDOM ANSWER]\n{rnd_answer}\n")
random_hall = 1 if hallucinated(rnd_answer) else 0
handle_random.remove()

# -----------------------
# Save outputs & a tiny delta bar
# -----------------------
lines = [
    f"Prompt: {QUESTION}",
    f"Baseline: {baseline_answer}  | hallucinated={baseline_hall}",
    f"Targeted (k={TOPK}): {tgt_answer}  | hallucinated={targeted_hall}",
    f"Random (k={TOPK}): {rnd_answer}  | hallucinated={random_hall}",
]
save_txt_lines(FIG_DIR / "ablation_text_outputs.txt", lines)
plot_ablation_delta(baseline_hall, targeted_hall, random_hall, FIG_DIR / "ablation_delta_bar.pdf")

print("[DONE] Figures saved to ./fig/:")
for f in ["attention_image_token_mass.pdf",
          "image_token_attention_bar.pdf",
          "ablation_delta_bar.pdf",
          "ablation_text_outputs.txt"]:
    print(" -", FIG_DIR / f)
