import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import joblib

# data = joblib.load("motion_data/demo_data.pkl")
# concat_data = np.concatenate(list(data.values()), axis=0)
# mean = np.mean(concat_data, axis=0)
# std = np.std(concat_data, axis=0)
# min_vals = np.min(concat_data, axis=0)
# max_vals = np.max(concat_data, axis=0)
# data['mean'] = mean
# data['std'] = std
# data['max'] = max_vals
# data['min'] = min_vals
# joblib.dump(data, "motion_data/demo_data_full.pkl")
# Example motion sequences (each row is a frame, each column a feature)
motion_sequence_1 = np.random.rand(100, 3)  # 100 frames, 3 features (e.g., joint angles)
motion_sequence_2 = np.random.rand(80, 3)   # 80 frames, 3 features

# Compute the distance matrix
D, wp = librosa.sequence.dtw(X=motion_sequence_1.T, Y=motion_sequence_2.T, metric='euclidean')
import pdb;pdb.set_trace()

# Plot the distance matrix with the warping path
plt.figure(figsize=(10, 7))
librosa.display.specshow(D, x_axis='frames', y_axis='frames', cmap='viridis')
plt.plot(wp[:, 1], wp[:, 0], label='Warping Path', color='r')
plt.title('Dynamic Time Warping (DTW)')
plt.xlabel('Motion Sequence 2 Frames')
plt.ylabel('Motion Sequence 1 Frames')
plt.legend()
plt.colorbar()
plt.show()
