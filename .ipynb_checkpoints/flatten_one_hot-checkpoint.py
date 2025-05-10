import pandas as pd
import numpy as np
import sys

# Usage check
if len(sys.argv) != 3:
    print("Usage: python flatten_one_hot.py <x_test.csv> <y_test.csv>")
    sys.exit(1)

x_path = sys.argv[1]
y_path = sys.argv[2]

# Load data
x_df = pd.read_csv(x_path)
y_df = pd.read_csv(y_path)

# Convert y from one-hot to single-column labels
if y_df.shape[1] > 1:
    y_labels = np.argmax(y_df.values, axis=1)
else:
    y_labels = y_df.values.squeeze()

# Save outputs
x_df.to_csv("x_test_single.csv", index=False)
pd.DataFrame(y_labels, columns=["label"]).to_csv("y_test_single.csv", index=False)

print("✅ Converted and saved:")
print("   - x_test_single.csv")
print("   - y_test_single.csv")
