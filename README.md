# Image Robustness Evaluation with Perturbations

This repository provides tools for **evaluating model robustness** under different image perturbations. It supports **loading local datasets**, applying augmentation/perturbations, visualizing results, and plotting model-wise performance comparisons.

---

## 📂 Project Structure

```
.
├── data/                     # Local dataset images
│   ├── class1/               # Example category
│   │   ├── img1.jpg
│   │   ├── img2.jpg
│   └── class2/
│       ├── img3.jpg
│       ├── img4.jpg
│
├── results/                  # Plots and saved evaluation results
│   ├── fig.png
│   ├── accuracy_table.csv
│
├── scripts/
│   ├── augment.py            # Data augmentation & perturbation functions
│   ├── evaluate.py           # Model evaluation pipeline
│   ├── plot_results.py       # Visualization (bar charts, grouped plots)
│
├── README.md                 # Project documentation
```

---

## 🚀 Features

* Load **local datasets** using PyTorch `ImageFolder`.
* Apply **augmentation and perturbations** (blur, noise, brightness, etc.).
* Evaluate **different models** on clean vs perturbed data.
* Generate **comparison plots**:

  * Accuracy table
  * Model-wise grouped bar charts
  * Perturbation-wise comparisons

---

## 📦 Requirements

Install dependencies:

```bash
pip install torch torchvision matplotlib pandas
```

---

## 📊 Usage

### 1. Prepare Dataset

Place your images under `./data/` following a class-folder structure:

```
data/
├── cats/
│   ├── cat1.jpg
│   ├── cat2.jpg
├── dogs/
│   ├── dog1.jpg
│   ├── dog2.jpg
```

### 2. Run Evaluation

```bash
python scripts/evaluate.py
```

This will:

* Load dataset
* Apply augmentations
* Evaluate models
* Save results in `./results/`

### 3. Plot Results

```bash
python scripts/plot_results.py
```

This will generate grouped bar charts and save as `results/fig.png`.

---

## 📷 Local Image Loading

You can directly load your **local dataset** like this:

```python
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

dataset = datasets.ImageFolder("./data", transform=transform)
```

---

## 📌 Notes

* Supported input: **local images**
* Image types: `.jpg`, `.png`, `.jpeg`
* Output: Accuracy table (`.csv`) and plots (`.png`)

---
<!-- 
## 🛠️ Next Steps

* Add more perturbation types
* Extend evaluation to larger benchmark datasets
* Compare across multiple pretrained models -->


