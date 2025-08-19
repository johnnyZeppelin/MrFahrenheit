import os
import re
import csv
import json
from pathlib import Path
from collections import defaultdict
from difflib import SequenceMatcher
from openai import OpenAI
from tqdm import tqdm
import matplotlib.pyplot as plt

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
FIGURES_DIR = Path("figures")
FIGURES_DIR.mkdir(exist_ok=True)
CACHE_FILE = Path("api_cache.json")

# -------------------
# Helper Functions
# -------------------
def normalize_brand(name: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", name.lower().strip())

def string_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def is_consistent(pred_text: str, brand: str, threshold=0.6) -> bool:
    pred_norm = normalize_brand(pred_text)
    gt_norm = normalize_brand(brand)
    return string_similarity(pred_norm, gt_norm) >= threshold

# Load cache if exists
if CACHE_FILE.exists():
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        api_cache = json.load(f)
else:
    api_cache = {}

def ask_model(client, model_name, image_path, task="brand"):
    """Send image to VLM with caching."""
    api_cache.setdefault(model_name, {}).setdefault(task, {})
    img_key = str(image_path)
    if img_key in api_cache[model_name][task]:
        return api_cache[model_name][task][img_key]

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
            messages=[{"role": "user", "content":[
                {"type":"text","text":question},
                {"type":"image_url","image_url":{"url":f"file://{image_path}"}}
            ]}]
        )
        result = completion.choices[0].message.content.strip()
    except Exception as e:
        result = f"ERROR: {e}"

    # Store in cache
    api_cache[model_name][task][img_key] = result
    return result

def save_cache():
    with open(CACHE_FILE, "w", encoding="utf-8") as f:
        json.dump(api_cache, f, ensure_ascii=False, indent=2)

def plot_metrics(metrics_dict, overall_acc, overall_ltc, title, filename):
    labels = list(metrics_dict.keys())
    accs = [metrics_dict[k]["accuracy"] for k in labels]
    ltcs = [metrics_dict[k]["consistency"] for k in labels]

    x = range(len(labels))
    fig, ax = plt.subplots(figsize=(12,6))
    ax.bar(x, accs, width=0.4, label="Accuracy", align="center")
    ax.bar([i+0.4 for i in x], ltcs, width=0.4, label="Logo-Text Consistency", align="center")
    ax.set_xticks([i+0.2 for i in x])
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_ylim(0,1)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, format="pdf")
    plt.close(fig)

def plot_multi_model(metrics_all_models, title, filename):
    models = list(metrics_all_models.keys())
    categories = list(next(iter(metrics_all_models.values())).keys())
    x = range(len(categories))
    fig, ax = plt.subplots(figsize=(12,6))
    width = 0.8 / len(models)
    for i, model in enumerate(models):
        vals = [metrics_all_models[model][cat]["accuracy"] for cat in categories]
        ax.bar([j + i*width for j in x], vals, width=width, label=model)
    ax.set_xticks([j + width*len(models)/2 for j in x])
    ax.set_xticklabels(categories, rotation=45, ha="right")
    ax.set_ylim(0,1)
    ax.set_ylabel("Accuracy")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(filename, format="pdf")
    plt.close(fig)

# -------------------
# Main Benchmark
# -------------------
if __name__=="__main__":
    client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=OPENROUTER_API_KEY)
    all_results = []
    metrics_per_category_all_models = {}
    metrics_per_trans_all_models = {}

    for model_name in vlm_models:
        print(f"\n[INFO] Evaluating {model_name} ...")
        total_imgs = sum(1 for _ in dataset_dir.rglob("*.jpg"))
        pbar_img = tqdm(total=total_imgs, desc=f"Images ({model_name})", ncols=100)

        results = []
        correct_classification = consistent_predictions = total = 0
        per_category_stats = defaultdict(lambda: {"total":0,"correct":0,"consistent":0})
        per_transformation_stats = defaultdict(lambda: {"total":0,"correct":0,"consistent":0})

        for transform_path in dataset_dir.iterdir():
            if not transform_path.is_dir(): continue
            transformation = transform_path.name
            for category_path in transform_path.iterdir():
                if not category_path.is_dir(): continue
                category = category_path.name
                for brand_path in category_path.iterdir():
                    if not brand_path.is_dir(): continue
                    brand = brand_path.name
                    img_files = list(brand_path.glob("*.jpg"))
                    for img_file in tqdm(img_files, desc=f"{transformation}/{category}/{brand}", leave=False, ncols=100):
                        total +=1
                        per_category_stats[category]["total"] +=1
                        per_transformation_stats[transformation]["total"] +=1

                        # Brand recognition
                        pred_brand_raw = ask_model(client, model_name, img_file, task="brand")
                        pred_brand = normalize_brand(pred_brand_raw)
                        gt_norm = normalize_brand(brand)
                        is_correct = (pred_brand == gt_norm)
                        if is_correct:
                            correct_classification +=1
                            per_category_stats[category]["correct"] +=1
                            per_transformation_stats[transformation]["correct"] +=1

                        # Logo-text consistency
                        pred_text_raw = ask_model(client, model_name, img_file, task="text")
                        is_ltc = is_consistent(pred_text_raw, brand)
                        if is_ltc:
                            consistent_predictions +=1
                            per_category_stats[category]["consistent"] +=1
                            per_transformation_stats[transformation]["consistent"] +=1

                        results.append({
                            "model": model_name,
                            "transformation": transformation,
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
                        pbar_img.update(1)

        pbar_img.close()
        save_cache()  # save cache after each model

        overall_acc = correct_classification / total if total>0 else 0
        overall_consistency = consistent_predictions / total if total>0 else 0

        per_category_metrics = {cat: {"accuracy": stats["correct"]/stats["total"], "consistency": stats["consistent"]/stats["total"]}
                                for cat, stats in per_category_stats.items()}
        per_trans_metrics = {t: {"accuracy": stats["correct"]/stats["total"], "consistency": stats["consistent"]/stats["total"]}
                             for t, stats in per_transformation_stats.items()}

        all_results.extend(results)
        metrics_per_category_all_models[model_name] = per_category_metrics
        metrics_per_trans_all_models[model_name] = per_trans_metrics

        print(f"[RESULT] {model_name} - Accuracy: {overall_acc:.2%}, Logo-Text Consistency: {overall_consistency:.2%}")

        # Save PDF plots per model
        model_fig_dir = FIGURES_DIR / model_name.replace("/", "_")
        model_fig_dir.mkdir(parents=True, exist_ok=True)
        plot_metrics(per_category_metrics, overall_acc, overall_consistency, f"{model_name} - Per Category", model_fig_dir / "per_category.pdf")
        plot_metrics(per_trans_metrics, overall_acc, overall_consistency, f"{model_name} - Per Transformation", model_fig_dir / "per_transformation.pdf")

    # Multi-model comparison plots
    plot_multi_model(metrics_per_category_all_models, "Multi-Model: Accuracy per Category", FIGURES_DIR / "multi_model_per_category.pdf")
    plot_multi_model(metrics_per_trans_all_models, "Multi-Model: Accuracy per Transformation", FIGURES_DIR / "multi_model_per_transformation.pdf")

    # Save CSV results
    out_csv = "vlm_benchmark_results.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_results[0].keys())
        writer.writeheader()
        writer.writerows(all_results)
    print(f"[INFO] Results saved to {out_csv}")
    print(f"[INFO] Figures saved to {FIGURES_DIR}")
