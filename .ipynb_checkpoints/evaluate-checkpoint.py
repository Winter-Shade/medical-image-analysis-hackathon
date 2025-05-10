import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import cv2

# Preprocessing function
def preprocess_general(img):
    if img.ndim == 3 and img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    elif img.ndim == 3 and img.shape[-1] == 1:
        img = img.squeeze()
    return img.astype(np.float32) / 255.0

# Command-line arguments
if len(sys.argv) < 3:
    print("Usage: python evaluate_model.py <x_test.csv> <y_test.csv> [--preprocess]")
    sys.exit(1)

x_test_path = sys.argv[1]
y_test_path = sys.argv[2]
apply_preprocess = "--preprocess" in sys.argv

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load test data
x_test = pd.read_csv(x_test_path).values.reshape(-1, 28, 28).astype(np.float32)
y_test = pd.read_csv(y_test_path).values.squeeze()

if y_test.ndim > 1:
    y_test = np.argmax(y_test, axis=1)

# Optional preprocessing
if apply_preprocess:
    print("🔧 Applying preprocessing...")
    x_test = np.array([preprocess_general(img) for img in x_test])

# Reshape and convert to tensors
x_test = x_test.reshape(-1, 1, 28, 28).astype(np.float32)
x_tensor = torch.tensor(x_test)
y_tensor = torch.tensor(y_test).long()

# DataLoader
test_dataset = TensorDataset(x_tensor, y_tensor)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# Load model
from model import CustomResNet18  # Replace with your actual model if different
model = CustomResNet18(num_classes=13)  # Update num_classes if needed
model.load_state_dict(torch.load("best_custom_resnet18.pth", map_location=device))
model.to(device)
model.eval()

# Evaluation function
def evaluate_model(model, loader, device):
    model.eval()
    y_true, y_pred, y_probs = [], [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())
            y_probs.extend(probs.cpu().numpy())

    accuracy = accuracy_score(y_true, y_pred)
    weighted_recall = recall_score(y_true, y_pred, average='weighted')
    macro_recall = recall_score(y_true, y_pred, average='macro')
    weighted_precision = precision_score(y_true, y_pred, average='weighted')
    macro_precision = precision_score(y_true, y_pred, average='macro')

    try:
        auc = roc_auc_score(y_true, y_probs, multi_class='ovr')
    except:
        auc = 0.0

    print(f"\n📊 Evaluation on Test Set:")
    print(f"Accuracy:           {accuracy:.4f}")
    print(f"Weighted Recall:    {weighted_recall:.4f}")
    print(f"Macro Recall:       {macro_recall:.4f}")
    print(f"Weighted Precision: {weighted_precision:.4f}")
    print(f"Macro Precision:    {macro_precision:.4f}")
    print(f"AUC Score:          {auc:.4f}")

# Run evaluation
if __name__ == "__main__":
    evaluate_model(model, test_loader, device)
