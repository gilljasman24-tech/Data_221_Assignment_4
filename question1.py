from sklearn.datasets import load_breast_cancer
import numpy as np

# Load dataset
data = load_breast_cancer()

# Feature matrix (X) and target vector (y)
X = data.data
y = data.target

# Report shapes
print("Shape of X:", X.shape)
print("Shape of y:", y.shape)

# Count samples in each class
unique, counts = np.unique(y, return_counts=True)

print("\nClass distribution:")
for u, c in zip(unique, counts):
    print(f"Class {u}: {c} samples")

# COMMENTS

# The dataset is slightly imbalanced since there are more benign cases than malignant ones.
# Class balance matters because a model can become biased toward the majority class.
# This can lead to misleading accuracy and poor detection of the minority class.