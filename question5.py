from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# Load dataset
data = load_breast_cancer()
X = data.data
y = data.target

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Decision Tree
tree_model = DecisionTreeClassifier(max_depth=3, random_state=42)
tree_model.fit(X_train, y_train)

tree_preds = tree_model.predict(X_test)
tree_cm = confusion_matrix(y_test, tree_preds)

print("Decision Tree Confusion Matrix:")
print(tree_cm)

# Neural Network
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

nn_model = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, random_state=42)
nn_model.fit(X_train_scaled, y_train)

nn_preds = nn_model.predict(X_test_scaled)
nn_cm = confusion_matrix(y_test, nn_preds)

print("\nNeural Network Confusion Matrix:")
print(nn_cm)

# COMMENTS

# The neural network performs slightly better since it makes fewer mistakes overall,
# especially in predicting malignant cases.
# I would prefer the neural network for this task because it has better performance.
# An advantage of the decision tree is that it is easy to understand and interpret.
# A limitation is that it may not capture more complex patterns.
# An advantage of the neural network is that it can model more complex relationships.
# A limitation is that it is harder to interpret.