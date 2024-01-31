# src/data_preprocessing/data_preprocessing.py
import os
import librosa
import numpy as np

def load_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    return y, sr

def extract_mfcc(segment, sr=44100, n_mfcc=13):
    mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
    
    # Print basic information about the shape of the array
    #print(f"MFCCs shape: {mfccs.shape}")
    
    return mfccs
def process_audio_file(file_path, label, segment_length=0.25, overlap=0.1, n_mfcc=13):
    audio, sr = load_audio(file_path)
    segment_samples = int(segment_length * sr)
    overlap_samples = int(overlap * sr)

    segments = []

    #print(f"Total audio length: {len(audio)}, Segment samples: {segment_samples}, Overlap samples: {overlap_samples}")

    for start in range(0, len(audio) - segment_samples + 1, overlap_samples):
        segment = audio[start:start + segment_samples]
        mfccs = extract_mfcc(segment, sr=sr, n_mfcc=n_mfcc)
        segments.append((mfccs, label))

    return segments


def process_all_audio_files(raw_data_dir, processed_data_dir, segment_length=0.25, overlap=0.1, n_mfcc=13):
    os.makedirs(processed_data_dir, exist_ok=True)

    for label in os.listdir(raw_data_dir):
        label_dir = os.path.join(raw_data_dir, label)
        if os.path.isdir(label_dir):
            label_data = []
            for file_name in os.listdir(label_dir):
                if file_name.endswith(".wav"):
                    file_path = os.path.join(label_dir, file_name)
                    label_data.extend(process_audio_file(file_path, label, segment_length, overlap, n_mfcc))
            
            # Print information about the processed data
            #print(f"Processed data for {label}: {len(label_data)} segments")
            np.save(os.path.join(processed_data_dir, f"{label}.npy"), label_data)

if __name__ == "__main__":
    raw_data_dir = "C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/data/raw_audio"
    processed_data_dir = "C:/Users/Tanmay Chhimwal/Documents/GitHub/FreeBox/data/processed_data"

    process_all_audio_files(raw_data_dir, processed_data_dir)
