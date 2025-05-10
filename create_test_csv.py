# create_test_csv.py

import os
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from medmnist import OCTMNIST, BreastMNIST, PneumoniaMNIST, RetinaMNIST, INFO

# Output directory
os.makedirs('split_data', exist_ok=True)

# Offsets to make labels unique across datasets
offsets = {
    'octmnist': 0,
    'breastmnist': len(INFO['octmnist']['label']),
    'pneumoniamnist': len(INFO['octmnist']['label']) + len(INFO['breastmnist']['label']),
    'retinamnist': len(INFO['octmnist']['label']) + len(INFO['breastmnist']['label']) + len(INFO['pneumoniamnist']['label']),
}

# General preprocessing
def preprocess_general(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze()
    return img.astype(np.float32) / 255.0

# Retina-specific preprocessing
def preprocess_retina(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    denoised = cv2.medianBlur(gray, 3)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(denoised).astype(np.float32) / 255.0

# Preprocessing loader
def load_and_preprocess(dataset_cls, offset, is_retina=False):
    X_test, y_test = [], []
    dataset = dataset_cls(split='test', download=True)
    for img, label in tqdm(zip(dataset.imgs, dataset.labels.squeeze()), total=len(dataset), desc=f"Processing {dataset_cls.__name__}"):
        label += offset
        proc_img = preprocess_retina(img) if is_retina else preprocess_general(img)
        X_test.append(proc_img)
        y_test.append(label)
    return X_test, y_test

# Load all datasets
X_test, y_test = [], []
for name, cls in zip(['octmnist', 'breastmnist', 'pneumoniamnist', 'retinamnist'],
                     [OCTMNIST, BreastMNIST, PneumoniaMNIST, RetinaMNIST]):
    is_retina = (name == 'retinamnist')
    Xt, yt = load_and_preprocess(cls, offsets[name], is_retina)
    X_test += Xt
    y_test += yt

X_test = np.array(X_test).reshape(-1, 28*28).astype(np.float32)
y_test = np.array(y_test).astype(np.int64)

# Save to CSV
pd.DataFrame(X_test).to_csv('split_data/x_test.csv', index=False)
pd.DataFrame(y_test).to_csv('split_data/y_test.csv', index=False)

print(f"\n✅ Saved x_test.csv with shape {X_test.shape}")
print(f"✅ Saved y_test.csv with shape {y_test.shape}")
