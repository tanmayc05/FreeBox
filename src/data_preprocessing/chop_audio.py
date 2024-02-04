#chop the beatboxing audio clip into different instruments
#take audio from full_length_audio folder, save the chopped audio in respective raw_audio folder

import librosa
import numpy as np
from keras.models import load_model

model = load_model("C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/beatbox_model")
labels = np.load("C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/labels.npy")

def process_audio_clip(file_path, model, labels):
    # Load audio clip
    audio, sr = librosa.load(file_path, sr=None)

    # Onset detection
    onsets = librosa.onset.onset_detect(y=audio, sr=sr, onset_envelope=librosa.onset.onset_strength(y=audio, sr=sr))

    # Set parameters
    n_mfcc = 16

    # Process the audio clip
    predicted_labels = []

    overlap = 0.5


    for onset in onsets:
        overlap_samples = int(overlap * sr)
        start = max(0, onset - n_mfcc // 2 - overlap_samples // 2)
        end = min(len(audio), onset + n_mfcc // 2 + overlap_samples // 2)

        segment = audio[start:end]
        mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)

        # Ensure the input shape matches the one used during training
        if mfccs.shape[1] < 22:
            # Pad the feature matrix with zeros to match the expected shape
            mfccs = np.pad(mfccs, ((0, 0), (0, 22 - mfccs.shape[1])))

        mfccs = np.expand_dims(mfccs, axis=0)  # Add batch dimension
        prediction = model.predict(mfccs)
        
        # Get the class index with the highest probability
        class_index = np.argmax(prediction, axis=1)[0]
        
        # Map class index to instrument label
        predicted_label = labels[class_index]

        # Append the predicted label for this onset
        predicted_labels.append(predicted_label)

    return predicted_labels

# Example usage:
audio_clip_path = "C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/data/full_length_audio/beatbox-loop-at-87-bpm-with-effects-72241.mp3"
predictions = process_audio_clip(audio_clip_path, model, labels)
print(predictions)


