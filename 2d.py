import numpy as np
import wfdb
import matplotlib.pyplot as plt

signal = wfdb.rdrecord('data/paf-prediction-challenge-database/n01')
annotations = wfdb.rdann('data/paf-prediction-challenge-database/n01', 'qrs')

index = annotations.sample.reshape(-1,1)+np.arange(1,50)

aligned = signal.p_signal[index]
plt.imshow(aligned[:,:,0], aspect='auto', cmap='jet', extent=[0,49,0,aligned.shape[0]])

plt.show()