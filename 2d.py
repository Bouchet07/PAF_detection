import wfdb
import matplotlib.pyplot as plt

signal = wfdb.rdrecord('data/paf-prediction-challenge-database/n01')
annotations = wfdb.rdann('data/paf-prediction-challenge-database/n01', 'qrs')
signal_align = signal.p_signal[annotations.sample - 10: annotations.sample + 10]
plt.plot(signal_align)
plt.scatter(10, signal_align[10], color='red', label='QRS Annotations')
plt.legend()
plt.title('ECG Signal with QRS Annotations')
plt.xlabel('Sample Number')
plt.ylabel('Amplitude')
plt.show()