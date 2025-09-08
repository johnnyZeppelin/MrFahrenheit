import torch, torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
# LLaVA-1.6 HF identifier varies by checkpoint; e.g. "liuhaotian/llava-v1.6-vicuna-7b-hf"
CKPT = "liuhaotian/llava-v1.6-vicuna-7b-hf"

# we need attentions from the LLM:
cfg = AutoConfig.from_pretrained(CKPT)
cfg.output_attentions = True
model = AutoModelForCausalLM.from_pretrained(
    CKPT, torch_dtype=torch.float16, device_map="auto", config=cfg
)
tok = AutoTokenizer.from_pretrained(CKPT, use_fast=False)
model.eval()

# ===========================================================================================

# --- Helper: register a forward hook on the projector to capture its output ---
proj_acts = {}

def register_projector_hook(model):
    """
    Find the projector module and attach a hook that stores its outputs.
    For LLaVA 1.6 HF, the projector is often accessible as model.visual_projector or model.model.vision_tower.mlp_proj.
    Fall back: print(model) and locate the Linear/MLP mapping from vision_dim -> hidden_size.
    """
    target = None
    # try common names
    for name, module in model.named_modules():
        if "vision_tower" in name and ("mlp" in name or "proj" in name) and hasattr(module, "forward"):
            target = module
    if target is None:
        # alternative path (some checkpoints expose visual_projector)
        for name, module in model.named_modules():
            if "visual_projector" in name:
                target = module
                break
    assert target is not None, "Could not find projector; inspect model.named_modules()."

    def hook(_m, _inp, out):
        # out: [B, N_img_tokens, d] or similar
        proj_acts["z"] = out.detach()

    h = target.register_forward_hook(hook)
    return h

proj_hook = register_projector_hook(model)

# ====================================================================================

from PIL import Image
import numpy as np

# minimal LLaVA VQA-style prompt
SYS = "You are a helpful multimodal assistant."
USER = "Does this logo contain any visible text? If yes, transcribe exactly; otherwise answer 'NO-TEXT'."
prompt = f"<s>[INST] <<SYS>>\n{SYS}\n<</SYS>>\n{USER} [/INST]"

# you will need the image processor paired with your checkpoint; here we assume model has a processor or you use LLaVA repo's processor
# For brevity, just show the conceptual call:
def encode_image_and_prompt(img: Image.Image, text: str):
    """
    Return input tensors that LLaVA expects, plus the positions of image tokens in the sequence.
    Your actual code will use the specific processor from the checkpoint (e.g., LlavaProcessor).
    """
    # pseudo:
    # inputs = processor(images=img, text=text, return_tensors="pt")
    # return inputs.to(model.device), image_token_start, image_token_end
    raise NotImplementedError

# Example usage:
# inputs, img_s, img_e = encode_image_and_prompt(Image.open("example.jpg"), prompt)
# with torch.no_grad():
#     out = model(**inputs, output_attentions=True)
# gens = model.generate(**inputs, max_new_tokens=64, do_sample=False, output_attentions=True)

