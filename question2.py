from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data (80/20 with stratification)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Decision Tree using entropy
model = DecisionTreeClassifier(criterion="entropy", random_state=42)
model.fit(X_train, y_train)

# Accuracy
train_acc = model.score(X_train, y_train)
test_acc = model.score(X_test, y_test)

print("Training Accuracy:", train_acc)
print("Test Accuracy:", test_acc)

# COMMENTS

# Entropy measures how mixed or uncertain the classes are at a node.
# The tree tries to reduce this uncertainty by choosing splits that separate the classes.
# The training accuracy is 1.0 while the test accuracy is lower (~0.91),
# which suggests the model is overfitting the training data.
# It is learning the training set very well but does not generalize as well to new data.