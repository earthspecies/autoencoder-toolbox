import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import librosa
import musdb

from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import glob
import re
from sklearn.model_selection import StratifiedKFold
import pandas as pd

def show_spec(signal, kernel_size=int(25/1000*16000), stride=int(10/1000*16000)):
	S = librosa.stft(signal, n_fft=kernel_size, hop_length=stride)
	D = librosa.amplitude_to_db(np.abs(S), ref=1.)
	plt.imshow(D, aspect='auto', cmap='magma')
	plt.gca().invert_yaxis()
	plt.axis('off')
	plt.show()

class MacaqueDataset(Dataset):
	def __init__(self, 
				 subset='train', 
				 sample_rate=24414, 
				 base_path='Data/Macaque', 
				 seed=42):
		self.base_path = base_path
		self.subset = subset
		self.sample_rate = sample_rate
		self.seed = seed
		random.seed(seed)
		
		self.files = sorted(glob.glob(f'{self.base_path}/*/*.wav'))
		self.audios = [self.fix_length(librosa.load(f, sr=None)[0], 
									   length=self.sample_rate) for f in self.files]
		self.labels = [re.split(r'/', f)[-2] for f in self.files]
		
		label_dict = {l:i for i,l in enumerate(np.unique(self.labels))}
		self.int_labels = [label_dict[l] for l in self.labels]
		
		x_train, y_train, x_test, y_test = self.train_test_split(self.audios,
																 self.int_labels)
		if subset == 'train':
			self.len = len(x_train)
			self.x = x_train
			self.y = y_train
		elif subset == 'test':
			self.len = len(x_test)
			self.x = x_test
			self.y = y_test

	def __len__(self):
		return self.len

	def __getitem__(self, idx):
		x = self.x[idx]
		x = torch.tensor(x).unsqueeze(dim=0)
		return x, x
	
	@staticmethod
	def fix_length(signal, length=24414):
		signal_length = len(signal)
		if signal_length < length:
			tail_length = random.randint(0, length-signal_length)
			head_length = length - (signal_length + tail_length)

			signal = np.concatenate([np.zeros(head_length), signal, np.zeros(tail_length)]).astype('float32')
		else:
			signal = signal[:length]
		return signal
	
	@staticmethod
	def train_test_split(X, Y, n_folds=5, seed=42):
		skf = StratifiedKFold(n_splits=5,
							  shuffle=True,
							  random_state=seed)
		train_split, test_split = list(skf.split(X, Y))[0]
		
		X_train = [X[i] for i in train_split]
		X_test = [X[i] for i in test_split]
		Y_train = [Y[i] for i in train_split]
		Y_test = [Y[i] for i in test_split]
		return X_train, Y_train, X_test, Y_test

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
    	"""Credit to Maddie Cusimano"""
		t = np.arange(start=0, stop=ramp_duration, step=1/sample_rate)
		off_ramp = 0.5*(1. + np.cos( (np.pi/ramp_duration)*t ))

		return off_ramp

class ESCDataset(Dataset):
	def __init__(self, subset='train', test_fold=5, data='Data/anno.pkl'):
		self.subset = subset
		self.test_fold = test_fold
		self.data = pd.read_pickle(data)
		
		train_data = self.data[self.data.fold != self.test_fold]
		test_data = self.data[self.data.fold == self.test_fold]
		
		if self.subset == 'train':
			self.x = train_data
		elif self.subset == 'test':
			self.x = test_data
		self.len = len(self.x)
		
	def __getitem__(self, idx):
		x = self.x.audio.iloc[idx]
		x = torch.Tensor(x).unsqueeze(dim=0)
		return x, x
	
	def __len__(self):
		return self.len
    
class MusDB18Dataset(torch.utils.data.Dataset):
	def __init__(self, 
				 n_samples=5000,
				 duration=4., 
				 sample_rate=44100,
				 elements=[
					 'vocals', 
					 'drums',
					 'bass',
					 'other'
				 ],
				 split='train',
				 root='Data',
				 seed=42):
		self.n_samples = n_samples
		self.duration = duration
		self.sample_rate = sample_rate
		self.elements = elements
		self.split = split
		self.root = root
		self.seed = seed
		random.seed(seed)
        
		self.mus = musdb.DB(root=self.root,
							subsets=self.split)
		self.len=n_samples
    
	def __len__(self):
		return self.len
    
	def __getitem__(self, idx):
		if self.split == 'test':
			random.seed(idx)
            
		track = random.choice(self.mus.tracks)
		track.chunk_duration = self.duration
		track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
		element = random.choice(self.elements)
		x = track.targets[element].audio.T[0]
		x = torch.Tensor(x).unsqueeze(dim=0)
		return x, x
    
class GeladaDataset(Dataset):
	def __init__(self, data_dir='Data/Geladas/annotations.pkl.gzip', subset='train', length=44100, target_label=None):
		self.table = pd.read_pickle(data_dir)
		self.subset = subset
		self.length = length
		self.target_label = target_label
		
		self.audios = self.table.call.apply(lambda x: librosa.util.fix_length(x, self.length)).to_list()
		if self.target_label is None:
			self.targets = [0 for _ in self.audios]
		else:
			self.labels = self.table[self.target_label].values
			self.label_dict = {l:i for i,l in enumerate(np.unique(self.labels))}
			self.int_labels = [self.label_dict[l] for l in self.labels]
			self.targets = self.int_labels
		
		x_train, y_train, x_test, y_test = self.train_test_split(self.audios,
																 self.targets)
		if subset == 'train':
			self.len = len(x_train)
			self.x = x_train
			self.y = y_train
		elif subset == 'test':
			self.len = len(x_test)
			self.x = x_test
			self.y = y_test
	
	def __len__(self):
		return self.len
	
	def __getitem__(self, idx):
		a = torch.tensor(self.x[idx]).unsqueeze(dim=0)
		if self.target_label is None:
			return a, a
		else:
			y_idx = self.y[idx]
			l = torch.LongTensor([y_idx]).squeeze()
			return a, a, l
	
	@staticmethod
	def train_test_split(X, Y, n_folds=5, seed=42):
		skf = StratifiedKFold(n_splits=5,
							  shuffle=True,
							  random_state=seed)
		train_split, test_split = list(skf.split(X, Y))[0]

		X_train = [X[i] for i in train_split]
		X_test = [X[i] for i in test_split]
		Y_train = [Y[i] for i in train_split]
		Y_test = [Y[i] for i in test_split]
		return X_train, Y_train, X_test, Y_test