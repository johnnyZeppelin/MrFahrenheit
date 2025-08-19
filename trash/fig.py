import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 构造 DataFrame
data = {
    "Model": [
        "gemini-2.5-pro", "gpt-5-mini", "qwen3-235b", "gpt-4o",
        "claude-sonnet-4", "grok-4", "llama-4-maverick", "glm-4.5v"
    ],
    "blur":  [99.1, 98.0, 97.3, 95.7, 94.4, 93.5, 91.0, 85.4],
    "flip-H":[98.8, 98.1, 97.2, 95.9, 94.3, 93.3, 91.2, 85.7],
    "flip-V":[98.9, 97.9, 97.1, 95.8, 94.5, 93.4, 91.1, 85.4],
    "invert-color":[99.1, 98.2, 97.4, 96.0, 94.6, 93.6, 91.3, 85.7],
    "occlusion": [98.7, 97.8, 97.0, 95.6, 94.2, 93.2, 90.9, 85.3],
    "rot-180":  [98.7, 97.9, 97.1, 95.7, 94.4, 93.4, 91.0, 85.5],
    "rot-90":   [98.6, 97.7, 96.9, 95.7, 94.2, 93.1, 91.0, 85.3],
    "rot-random":[98.7, 97.8, 97.0, 95.8, 94.3, 93.3, 91.2, 85.6],
    "sharpen":  [99.1, 98.2, 97.4, 95.9, 94.5, 93.5, 91.2, 85.7],
    "Total":    [98.85, 97.95, 97.15, 95.79, 94.37, 93.36, 91.09, 85.50]
}
df = pd.DataFrame(data)

# 去掉 Total 列，并 melt 为长表格
df_long = df.drop(columns=["Total"]).melt(id_vars="Model", var_name="Perturbation", value_name="Accuracy")

# 分组柱状图绘制
models = df["Model"].values
perturbations = df_long["Perturbation"].unique()
x = np.arange(len(models))  # 横坐标位置
width = 0.1  # 柱子宽度

plt.figure(figsize=(14,6))
for i, perturbation in enumerate(perturbations):
    subset = df_long[df_long["Perturbation"] == perturbation]
    plt.bar(x + i*width, subset["Accuracy"], width=width, label=perturbation)

plt.ylabel("Accuracy (%)")
plt.title("Model Robustness under Different Perturbations")
plt.xticks(x + width*(len(perturbations)/2-0.5), models, rotation=45, ha="right")
plt.legend(title="Perturbation", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# 保存图片
plt.savefig("fig.png", dpi=300)
plt.show()
