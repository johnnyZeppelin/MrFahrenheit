# Image Robustness Evaluation with Perturbations

This repository provides tools for **evaluating model robustness** under different image perturbations. It supports **loading local datasets**, applying augmentation/perturbations, visualizing results, and plotting model-wise performance comparisons.

---

## 📂 Project Structure
This is the aug_dataset generated after running augment.py.

```
.
aug_dataset/
│
├── rotation/
│   └── LogoDet-3K/
│       ├── clothes/
│       │   ├── nike/
│       │   │   ├── img001.jpg
│       │   │   ├── img002.jpg
│       │   │   └── ...
│       │   ├── adidas/
│       │   │   ├── img101.jpg
│       │   │   └── ...
│       │   └── ...
│       └── food/
│           ├── starbucks/
│           │   ├── img201.jpg
│           │   └── ...
│           └── ...
│
├── blur/
│   └── LogoDet-3K/
│       ├── clothes/
│       │   ├── nike/
│       │   │   ├── img001.jpg
│       │   │   └── ...
│       │   └── adidas/
│       │       ├── img101.jpg
│       │       └── ...
│       └── ...
│
├── noise/
│   └── LogoDet-3K/
│       ├── clothes/
│       │   ├── nike/
│       │   │   ├── img001.jpg
│       │   │   └── ...
│       └── ...
│
└── ...
```

---

## 🚀 Features

* Load **local datasets** of any similar logo datasets.
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
bash env.sh
```

---

## 📊 Usage

### 1. Prepare Dataset

Place your images under `./data/` following a class-folder structure:

```
data/
├── <dataset name>/
```

```bash
python augment.py
```

This will:

* Load dataset
* Apply augmentations

### 2. Run Evaluation

```bash
python evaluate.py
```

This will:

* Evaluate models
* Save results in `./figures/`

### 3. Plot Results

```bash
python evaluate.py
```

This will also generate grouped bar charts and save as `./figures/<model name>.pdf`.

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

<!-- --- -->
<!-- 
## 🛠️ Next Steps

* Add more perturbation types
* Extend evaluation to larger benchmark datasets
* Compare across multiple pretrained models -->


