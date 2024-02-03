import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

# Load processed data
processed_data_dir = "C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/data/processed_data"
labels = os.listdir(processed_data_dir)
num_classes = len(labels)

# Load data from .npy files
data = []
for label in labels:
    file_path = os.path.join(processed_data_dir, label, f"{label}.npy")
    label_data = np.load(file_path, allow_pickle=True)
    data.extend(label_data)

np.save("C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/labels.npy", labels)

# Split data into features (X) and labels (y)
X = np.array([item[0] for item in data])
y = np.array([labels.index(item[1]) for item in data])

# Print the shape of the data
print("Shape of X:", X.shape)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(X_train[0].shape[0], X_train[0].shape[1])),
    tf.keras.layers.Conv1D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling1D(2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

save_directory = "C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/"
os.makedirs(save_directory, exist_ok=True)
model.save(os.path.join(save_directory, "beatbox_model"))

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test)
#print(f'Test accuracy: {test_acc}')
