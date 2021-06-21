import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np
from itertools import permutations, product

def weights_init(m):
	if (type(m) == nn.Conv1d) or (type(m) == nn.ConvTranspose1d):
		torch.nn.init.xavier_uniform_(m.weight)
        
def id_mapper(y):
	try:
		ids = np.unique(y.numpy()).tolist()
		id_dict = {e:i for i,e in enumerate(ids)}
		for i in range(y.size(0)):
			y[i] = id_dict[y[i].item()]
	except AttributeError:
		ids = np.unique(y).tolist()
		id_dict = {e:i for i,e in enumerate(ids)}
		for i in range(len(y)):
			y[i] = id_dict[y[i]]
	return y

def stft_mag_transform(x, stft):
	assert stft.coords == 'polar', 'Transform uses magnitude'
	_, mag = stft(x)
	return mag

def vae_recon_wrapper(func):
	def inner_args(*args, **kwargs):
		x_hat = args[0]
		x = args[1]
		return func(x_hat, x, **kwargs)
	return inner_args

def vae_kld_wrapper(func):
	def inner_args(*args):
		mu = args[2]
		logvar = args[3]
		return func(mu, logvar)
	return inner_args

def vq_vae_loss_wrapper(func):
	def inner_args(*args):
		l = args[3]
		return func(l)
	return inner_args

def vae_metric_wrapper(func):
	def inner_args(*args, **kwargs):
		x_hat = args[0]
		data = args[-1]
		return func(x_hat, data, **kwargs)
	return inner_args

def seed_everything(seed=111):
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
    
def logistic(x, _max, _min, k, x0):
	y = (_max - _min) / (1 + np.exp(-k * (x - x0))) + _min
	return y