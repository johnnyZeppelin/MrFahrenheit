import os
import re
import csv
from pathlib import Path
from openai import OpenAI
# from tqdm import tqdm

# -------------------
# CONFIG
# -------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

# Models to benchmark
vlm_models = [
    "z-ai/glm-4.5v",
    "openai/gpt-5-mini",
    # "openai/gpt-4o-mini",
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    # "anthropic/claude-3.5-sonnet",
    "meta-llama/llama-4-maverick",
    # "meta-llama/llama-3.2-vision",
    "x-ai/grok-4",
    # "google/gemini-1.5-flash"
    "qwen/qwen3-235b-a22b-2507"
]

# Dataset path
dataset_dir = Path("aug_dataset")
site_url = "https://lsfjohn.asia"  # "https://example.com"
site_name = "VLM Benchmark"

# -------------------
# Helper Functions
# -------------------
def normalize_brand(name: str) -> str:
    """Lowercase, remove punctuation and extra spaces."""
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())

def ask_model(client, model_name, image_path):
    """Send image to a VLM and get predicted brand."""
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": site_url,
                "X-Title": site_name,
            },
            model=model_name,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What brand or logo do you see in this image? Only answer with the brand name."},
                        {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                    ]
                }
            ]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# -------------------
# Main Loop
# -------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=OPENROUTER_API_KEY,
)

results = []

for model_name in vlm_models:
    print(f"\n[INFO] Evaluating {model_name}...")
    correct = 0
    total = 0

    for img_path in tqdm(dataset_dir.rglob("*.jpg")):
        # Ground truth from folder structure: aug_dataset/<transform>/<category>/<brand>/<file>.jpg
        parts = img_path.parts
        brand = parts[-2]  # second last folder is brand name

        gt = normalize_brand(brand)
        pred_raw = ask_model(client, model_name, img_path)
        pred = normalize_brand(pred_raw)

        is_correct = (gt == pred)
        if is_correct:
            correct += 1
        total += 1

        results.append({
            "model": model_name,
            "transformation": parts[1],
            "category": parts[2],
            "brand": brand,
            "image": img_path.name,
            "ground_truth": gt,
            "prediction_raw": pred_raw,
            "prediction_norm": pred,
            "correct": is_correct
        })

    accuracy = correct / total if total > 0 else 0
    print(f"[RESULT] {model_name} Accuracy: {accuracy:.2%}")

# -------------------
# Save Results
# -------------------
out_csv = "vlm_benchmark_results.csv"
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

print(f"[INFO] Results saved to {out_csv}")
