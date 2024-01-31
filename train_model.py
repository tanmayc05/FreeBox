# train_model.py

import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

def load_processed_data(data_dir):
    data = []
    labels = []

    for label in os.listdir(data_dir):
        if label.endswith('.npy'):
            label_data = np.load(os.path.join(data_dir, label), allow_pickle=True)
            data.extend(label_data)
            labels.extend([label[:-4]] * len(label_data))

    data = np.array(data)
    labels = np.array(labels)

    return data, labels

def preprocess_data(data, labels):
    if len(data) == 0:
        raise ValueError("The dataset is empty. Check the data array.")

    # Assuming data is a 2D array (num_samples, num_mfcc_coefficients)
    num_samples, num_mfcc_coefficients = data.shape

    # Add a new dimension to represent channels (1 for grayscale)
    data = data.reshape((num_samples, num_mfcc_coefficients, 1))

    # Check if the number of samples is greater than the test_size ratio
    if num_samples <= int(0.2 * num_samples):
        raise ValueError("The number of samples is too small. Adjust test_size or provide more data.")

    # Shuffle and split the data
    indices = np.arange(num_samples)
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]

    # Encode labels
    label_encoder = LabelEncoder()
    labels_encoded = label_encoder.fit_transform(labels)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(data, labels_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder



def build_model(input_shape, num_classes):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32):
    # Train the model
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=batch_size)

    # Evaluate the model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f'/nTest accuracy: {test_acc}')

    return model

def save_model(model, label_encoder, model_dir='models', model_filename='beatbox_model.h5', encoder_filename='label_encoder.npy'):
    # Save the trained model
    model_path = os.path.join(model_dir, model_filename)
    model.save(model_path)

    # Save the label encoder
    encoder_path = os.path.join(model_dir, encoder_filename)
    np.save(encoder_path, label_encoder.classes_)

    print(f'Model saved to {model_path}')
    print(f'Label encoder saved to {encoder_path}')

if __name__ == "__main__":
    # Set your data directory
    data_dir = 'C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/data/processed_data'

    # Load and preprocess data
    data, labels = load_processed_data(data_dir)
    X_train, X_test, y_train, y_test, label_encoder = preprocess_data(data, labels)

    # Build the model
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    num_classes = len(np.unique(labels))
    model = build_model(input_shape, num_classes)

    # Train the model
    trained_model = train_model(model, X_train, y_train, X_test, y_test, epochs=10, batch_size=32)

    # Save the trained model and label encoder
    save_model(trained_model, label_encoder)
