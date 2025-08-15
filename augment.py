import os
import json
import random
import cv2
import kagglehub
import xml.etree.ElementTree as ET
from tqdm import tqdm
from pathlib import Path
import albumentations as A

# ----------------------
# Step 1: Download dataset
# ----------------------
print("[INFO] Downloading LogoDet-3K dataset...")
dataset_path = Path(kagglehub.dataset_download("lyly99/logodet3k"))
print("Path to dataset files:", dataset_path)

output_dir = Path("aug_dataset")
output_dir.mkdir(parents=True, exist_ok=True)

log_file = output_dir / "transform_log.json"
transform_records = []

# ----------------------
# Step 2: Define transformations
# ----------------------
transformations = {
    "rotate_90": A.Rotate(limit=(90, 90), p=1.0),
    "rotate_180": A.Rotate(limit=(180, 180), p=1.0),
    "rotate_random": A.Rotate(limit=45, p=1.0),
    "flip_horizontal": A.HorizontalFlip(p=1.0),
    "flip_vertical": A.VerticalFlip(p=1.0),
    "invert_color": A.InvertImg(p=1.0),
    "blur": A.GaussianBlur(blur_limit=7, p=1.0),
    "sharpen": A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=1.0),
    "occlusion": A.CoarseDropout(max_holes=3, max_height=0.3, max_width=0.3, fill_value=0, p=1.0),
}

# ----------------------
# Step 3: Helper - parse Pascal VOC XML
# ----------------------
def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    bboxes = []
    for obj in root.findall("object"):
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)
        bboxes.append((xmin, ymin, xmax, ymax))
    return bboxes

# ----------------------
# Step 4: Walk dataset
# ----------------------
print("[INFO] Applying transformations...")
for img_path in tqdm(dataset_path.rglob("*.jpg")):
    xml_path = img_path.with_suffix(".xml")
    if not xml_path.exists():
        continue

    img = cv2.imread(str(img_path))
    if img is None:
        continue

    bboxes = parse_voc_xml(xml_path)
    if not bboxes:
        continue

    # Category & brand are the first two subfolders under dataset_path
    rel_parts = img_path.relative_to(dataset_path).parts
    category = rel_parts[0]
    brand = rel_parts[1]

    for tname, tfunc in transformations.items():
        aug_img = img.copy()

        for (xmin, ymin, xmax, ymax) in bboxes:
            logo_crop = aug_img[ymin:ymax, xmin:xmax]
            if logo_crop.size == 0:
                continue
            transformed = tfunc(image=logo_crop)['image']
            aug_img[ymin:ymax, xmin:xmax] = transformed

        # Save augmented image preserving folder structure
        save_dir = output_dir / tname / category / brand
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), aug_img)

        # Log transformation
        transform_records.append({
            "original_file": str(img_path),
            "augmented_file": str(save_path),
            "transformation": tname,
            "category": category,
            "brand": brand
        })

# ----------------------
# Step 5: Save log
# ----------------------
with open(log_file, 'w') as f:
    json.dump(transform_records, f, indent=2)

print(f"[INFO] Augmentation complete. Results saved in '{output_dir}'.")
print(f"[INFO] Transformation log saved to '{log_file}'.")
