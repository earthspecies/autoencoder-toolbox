import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import librosa

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np

def show_spec(signal, kernel_size=int(25/1000*16000), stride=int(10/1000*16000)):
	S = librosa.stft(signal, n_fft=kernel_size, hop_length=stride)
	D = librosa.amplitude_to_db(np.abs(S), ref=1.)
	plt.imshow(D, aspect='auto', cmap='magma')
	plt.gca().invert_yaxis()
	plt.axis('off')
	plt.show()

class ChirpDataset(Dataset):
	def __init__(self, 
				 n_samples=1000, 
				 subset='train',
				 min_freq=50,
				 min_delta=1000,
				 max_freq=8000,
				 max_delta=5000,
				 duration=1,
				 sample_rate=16000,
				 ramp = False,
				 seed=42):
		self.n_samples = n_samples
		self.subset = subset

		self.min_freq = min_freq
		self.min_delta = min_delta
		self.max_freq = max_freq
		self.max_delta = max_delta
		self.duration = duration
		self.sample_rate = sample_rate
		self.ramp = ramp

		random.seed(seed)

	def __len__(self):
		return self.n_samples

	def __getitem__(self, idx):
		if self.subset == 'test':
			random.seed(idx)
		fmin = random.uniform(self.min_freq, self.min_freq + self.min_delta)
		fmax = random.uniform(self.max_freq, self.max_freq - self.max_delta)
		if idx % 2 == 0:
			fmin, fmax = fmax, fmin

		chirp = librosa.chirp(fmin=fmin,
							  fmax=fmax,
							  sr=self.sample_rate,
							  duration=self.duration)
		if self.ramp:
			off_ramp = self.hann_ramp(self.sample_rate)
			on_ramp = off_ramp[::-1]
			ramp_length = len(off_ramp)
			chirp[-ramp_length:] = chirp[-ramp_length:] * off_ramp
			chirp[:ramp_length] = chirp[:ramp_length] * on_ramp
		x = torch.Tensor(chirp).unsqueeze(	dim=0)
		return x, x

	@staticmethod
	def hann_ramp(sample_rate, ramp_duration = 0.005):
    
		t = np.arange(start=0, stop=ramp_duration, step=1/sample_rate)
		off_ramp = 0.5*(1. + np.cos( (np.pi/ramp_duration)*t ))

		return off_ramp
