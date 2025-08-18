import os
import re
import csv
from pathlib import Path
from difflib import SequenceMatcher
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from openai import OpenAI
from tqdm import tqdm

# -------------------
# CONFIG
# -------------------
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("Please set the OPENROUTER_API_KEY environment variable.")

vlm_models = [
    "z-ai/glm-4.5v",
    "openai/gpt-5-mini",
    "openai/gpt-4o",
    "google/gemini-2.5-pro",
    "anthropic/claude-sonnet-4",
    "meta-llama/llama-4-maverick",
    "x-ai/grok-4",
    "qwen/qwen3-235b-a22b-2507"
]

dataset_dir = Path("aug_dataset")
site_url = "https://lsfjohn.asia"
site_name = "VLM Benchmark"

# -------------------
# Helper Functions
# -------------------
def normalize_brand(name: str) -> str:
    """Lowercase, remove punctuation and spaces."""
    return re.sub(r"[^a-z0-9]", "", name.lower().strip())

def string_similarity(a: str, b: str) -> float:
    """Fuzzy string similarity between [0,1]."""
    return SequenceMatcher(None, a, b).ratio()

def is_consistent(pred_text: str, gt_text: str, threshold: float = 0.8) -> bool:
    pred = normalize_brand(pred_text)
    gt = normalize_brand(gt_text)
    if not pred or not gt:
        return False
    return string_similarity(pred, gt) >= threshold

def ask_model(client, model_name, image_path, task="brand"):
    """Send image to VLM and get prediction based on task."""
    if task == "brand":
        question = "What brand or logo do you see in this image? Only answer with the brand name."
    elif task == "text":
        question = "Read any visible text in this logo image. Answer only with the text content."
    else:
        raise ValueError("Invalid task type")

    try:
        completion = client.chat.completions.create(
            extra_headers={"HTTP-Referer": site_url, "X-Title": site_name},
            model=model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    {"type": "image_url", "image_url": {"url": f"file://{image_path}"}}
                ]
            }]
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"

# -------------------
# Evaluation
# -------------------
def evaluate_model(client, model_name, dataset_root, threshold=0.8):
    total = 0
    correct_classification = 0
    consistent_predictions = 0
    per_category_stats = defaultdict(lambda: {"total":0,"correct":0,"consistent":0})
    results = []

    for category_path in tqdm(list(dataset_root.iterdir()), desc=f"Categories ({model_name})"):
        if not category_path.is_dir():
            continue
        category = category_path.name
        for brand_path in category_path.iterdir():
            if not brand_path.is_dir():
                continue
            brand = brand_path.name
            for img_file in brand_path.glob("*.jpg"):
                total +=1
                per_category_stats[category]["total"] += 1

                # --- Brand recognition ---
                pred_brand_raw = ask_model(client, model_name, img_file, task="brand")
                pred_brand = normalize_brand(pred_brand_raw)
                gt_norm = normalize_brand(brand)
                is_correct = (pred_brand == gt_norm)
                if is_correct:
                    correct_classification +=1
                    per_category_stats[category]["correct"] +=1

                # --- Logo-text consistency ---
                pred_text_raw = ask_model(client, model_name, img_file, task="text")
                is_ltc = is_consistent(pred_text_raw, brand, threshold)
                if is_ltc:
                    consistent_predictions +=1
                    per_category_stats[category]["consistent"] +=1

                # --- Record per image ---
                results.append({
                    "model": model_name,
                    "category": category,
                    "brand": brand,
                    "image": img_file.name,
                    "ground_truth": gt_norm,
                    "prediction_raw": pred_brand_raw,
                    "prediction_norm": pred_brand,
                    "correct": is_correct,
                    "predicted_text_raw": pred_text_raw,
                    "logo_text_consistency": is_ltc
                })

    # overall metrics
    overall_acc = correct_classification / total if total > 0 else 0
    overall_consistency = consistent_predictions / total if total > 0 else 0

    # per-category metrics
    per_category_metrics = {}
    for cat, stats in per_category_stats.items():
        total_cat = stats["total"]
        per_category_metrics[cat] = {
            "accuracy": stats["correct"]/total_cat if total_cat>0 else 0,
            "consistency": stats["consistent"]/total_cat if total_cat>0 else 0
        }

    return results, overall_acc, overall_consistency, per_category_metrics

# -------------------
# Plotting
# -------------------
def plot_vlm_metrics(per_category_metrics, overall_acc, overall_consistency, title="VLM Evaluation"):
    categories = list(per_category_metrics.keys())
    accs = [per_category_metrics[c]["accuracy"] for c in categories]
    ltc = [per_category_metrics[c]["consistency"] for c in categories]

    sns.set(style="whitegrid")
    plt.figure(figsize=(10,6))
    bar_width = 0.4
    x = range(len(categories))
    plt.bar([i - bar_width/2 for i in x], accs, width=bar_width, label="Accuracy", color="#4c72b0")
    plt.bar([i + bar_width/2 for i in x], ltc, width=bar_width, label="Logo-Text Consistency", color="#55a868")
    plt.xticks(x, categories, rotation=45, ha='right')
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.title(title)
    plt.legend()
    plt.text(len(categories)-0.5,0.95,f"Overall Accuracy: {overall_acc:.2%}\nOverall LTC: {overall_consistency:.2%}",
             fontsize=10,bbox=dict(facecolor="white", alpha=0.7, edgecolor="gray"))
    plt.tight_layout()
    plt.show()

def plot_multi_vlm_metrics(all_models_metrics, title="VLM Benchmark Comparison"):
    categories = list(next(iter(all_models_metrics.values())).keys())
    n_models = len(all_models_metrics)
    n_categories = len(categories)

    sns.set(style="whitegrid")
    plt.figure(figsize=(12,6))
    bar_width = 0.35
    x = range(n_categories)

    colors_acc = sns.color_palette("Blues", n_models)
    colors_ltc = sns.color_palette("Greens", n_models)

    for idx, (model_name, metrics) in enumerate(all_models_metrics.items()):
        accs = [metrics[c]["accuracy"] for c in categories]
        ltc = [metrics[c]["consistency"] for c in categories]
        offset = (idx - n_models/2) * bar_width + bar_width/2
        plt.bar([i + offset for i in x], accs, width=bar_width, color=colors_acc[idx], label=f"{model_name} Acc")
        plt.bar([i + offset for i in x], ltc, width=bar_width/2, color=colors_ltc[idx], alpha=0.5, label=f"{model_name} LTC")

    plt.xticks(x, categories, rotation=45, ha='right')
    plt.ylabel("Score")
    plt.ylim(0,1)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left', fontsize=9)
    plt.tight_layout()
    plt.show()

# -------------------
# Main Script
# -------------------
if __name__=="__main__":
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    all_results = []
    all_models_metrics = {}

    for model_name in vlm_models:
        print(f"\n[INFO] Evaluating {model_name} ...")
        results, overall_acc, overall_consistency, per_cat_metrics = evaluate_model(client, model_name, dataset_dir)
        all_results.extend(results)
        all_models_metrics[model_name] = per_cat_metrics
        print(f"[RESULT] {model_name} - Accuracy: {overall_acc:.2%}, Logo-Text Consistency: {overall_consistency:.2%}")
        plot_vlm_metrics(per_cat_metrics, overall_acc, overall_consistency, title=f"{model_name} Evaluation")

    # Multi-model comparative figure
    plot_multi_vlm_metrics(all_models_metrics, title="VLM Benchmark: Accuracy & Logo-Text Consistency per Category")

    # Save CSV
    out_csv = "vlm_benchmark_results.csv"
    if all_results:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
            writer.writeheader()
            writer.writerows(all_results)
        print(f"[INFO] Results saved to {out_csv}")
