import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Load dataset
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()

# Normalize
X_train = X_train / 255.0
X_test = X_test / 255.0

# Reshape
X_train = X_train.reshape(-1, 28, 28, 1)
X_test = X_test.reshape(-1, 28, 28, 1)

# Build CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Train quickly
model.fit(X_train, y_train, epochs=15, verbose=0)

# Predictions
y_pred = np.argmax(model.predict(X_test), axis=1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# Find misclassified indices
misclassified = np.where(y_test != y_pred)[0]

# Show 3 misclassified images
print("\nMisclassified examples:")

for i in range(3):
    idx = misclassified[i]
    plt.imshow(X_test[idx].reshape(28,28), cmap='gray')
    plt.title(f"True: {class_names[y_test[idx]]}, Pred: {class_names[y_pred[idx]]}")
    plt.axis('off')
    plt.show()

# COMMENTS

# One pattern is that similar clothing items like sneakers and sandals
# or shirts and pullovers are often confused with each other.
# This happens because these items can have similar shapes or features in the images.
# A possible way to improve performance is to use a deeper CNN model
# or train for more epochs to help the model learn more detailed patterns.