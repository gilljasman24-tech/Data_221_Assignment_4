from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data (80/20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train constrained Decision Tree (limit depth)
model = DecisionTreeClassifier(max_depth=3, random_state=42)
model.fit(X_train, y_train)

# Accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# Feature importance
importances = model.feature_importances_
feature_names = data.feature_names

# Get top 5 features
indices = np.argsort(importances)[::-1][:5]

print("\nTop 5 Important Features:")
for i in indices:
    print(f"{feature_names[i]}: {importances[i]:.4f}")

# COMMENTS

# Limiting the depth of the tree reduces overfitting by stopping the model
# from becoming too complex.
# Compared to the previous model, the training accuracy is lower,
# but the test accuracy is slightly higher, which suggests better generalization.
# Feature importance shows which variables the model relies on most.
# This helps us understand how the model is making decisions, making it easier to interpret.