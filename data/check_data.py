import numpy as np

# Replace 'your_file.npy' with the actual path to your .npy file
file_path = 'data\processed_data\clap.npy'

# Load the data from the .npy file
loaded_data = np.load(file_path, allow_pickle=True)

# Print the shape and contents of the loaded array
print("Shape of the loaded data:", loaded_data.shape)
print("Contents of the loaded data:")
print(loaded_data)
