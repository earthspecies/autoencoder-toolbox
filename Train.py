import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import json

from Datasets import *
from Autoencoder import Autoencoder
from Encoders import *
from Decoders import *
from PyFireUpdate import Trainer
from Utils import *

from Losses import *
from Metrics import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('-c', '--config', type=str,
						help='JSON file for configuration')

	args = parser.parse_args()

	with open(f'Configs/{args.config}') as f:
		data = f.read()
	config = json.loads(data)

	global dataset_config
	dataset_config = config['dataset_config']

	global encoder_config
	encoder_config = config['model_config']['encoder_config']
	global decoder_config
	decoder_config = config['model_config']['decoder_config']
	global autoencoder_config
	autoencoder_config = config['model_config']

	global learning_params
	learning_params = config['learning_params']
	global trainer_params
	trainer_params = config['trainer_params']

	train_set = ChirpDataset(subset='train',
							 n_samples=dataset_config['n_train_samples'],
							 **dataset_config['signal_params'])
	val_set = ChirpDataset(subset='test',
						   n_samples=dataset_config['n_test_samples'],
						   **dataset_config['signal_params'])
	train_loader = DataLoader(train_set,
							  batch_size=learning_params['batch_size'],
							  shuffle=True)
	val_loader = DataLoader(val_set,
							batch_size=learning_params['batch_size'],
							shuffle=False)

	if encoder_config['model_name'] == 'LightweightConv':
		encoder = LightweightConvEncoder(**encoder_config['model_params'])
	elif encoder_config['model_name'] == 'HeavyConv':
		encoder = HeavyEncoder()
		decoder = HeavyDecoder(encoder)

	if decoder_config['model_name'] == 'LightweightConv':
		decoder = LightweightConvDecoder(**decoder_config['model_params'])


	autoencoder = Autoencoder(encoder, 
							  decoder,
							  **autoencoder_config['model_params'])

	optimizer = torch.optim.Adam(autoencoder.parameters(), 
								 lr=learning_params['learning_rate'])
	scheduler = None#torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

	stft = STFT(kernel_size=encoder_config['model_params']['nfft'],
				stride=encoder_config['model_params']['hop'],
				coords='polar',
				dB=False)

	loss_fx = lambda x, y: perceptual_loss(x, y, stft=stft)
	loss_func = {'Perceptual_Loss': loss_fx}

	metric_fx = si_sdr
	metric_func = {'SI_SDR': metric_fx}

	if trainer_params['device'] == 'cuda':
		if torch.cuda.is_available():
			pass
		else:
			print('CUDA not available. Switching to CPU.')
			trainer_params['device'] = 'cpu'

	trainer = Trainer(autoencoder, optimizer, 
					  scheduler=scheduler,
					  loss_func=loss_func,
					  metric_func=metric_func,
					  verbose=trainer_params['verbose'],
					  device=trainer_params['device'],
					  dest=trainer_params['dest'],
					  **trainer_params['params'])
	trainer.fit(train_loader, 
				val_loader,
				learning_params['epochs'])
	trainer.save_model()

