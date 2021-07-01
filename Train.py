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
from PyFire import Trainer
from Utils import *

from Losses import *
from Metrics import *
import matplotlib.pyplot as plt

if __name__ == '__main__':

	seed_everything()
    
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

	if dataset_config['name'] == 'Chirp':
		train_set = ChirpDataset(subset='train',
								 n_samples=dataset_config['n_train_samples'],
								 **dataset_config['signal_params'])
		val_set = ChirpDataset(subset='test',
							   n_samples=dataset_config['n_test_samples'],
							   **dataset_config['signal_params'])
	elif dataset_config['name'] == 'Macaque':
		train_set = MacaqueDataset(subset='train')
		val_set = MacaqueDataset(subset='test')
	elif dataset_config['name'] == 'ESC':
		train_set = ESCDataset(subset='train')
		val_set = ESCDataset(subset='test')
	elif dataset_config['name'] == 'MusDB18':
		train_set = MusDB18Dataset(split='train',
								 n_samples=dataset_config['n_train_samples'],
								 **dataset_config['signal_params'])
		val_set = MusDB18Dataset(split='test',
								 n_samples=dataset_config['n_test_samples'],
								 **dataset_config['signal_params'])
	elif dataset_config['name'] == 'Geladas':
		train_set = GeladaDataset(subset='train')
		val_set = GeladaDataset(subset='test')
        
	train_loader = DataLoader(train_set,
							  batch_size=learning_params['batch_size'],
							  shuffle=True)
	val_loader = DataLoader(val_set,
							batch_size=learning_params['batch_size'],
							shuffle=False)

	if encoder_config['model_name'] == 'ToyConv':
		encoder = ToyConvEncoder(**encoder_config['model_params'])
	elif encoder_config['model_name'] == 'Conv':
		encoder = ConvEncoder(**encoder_config['model_params'])
	if encoder_config['model_name'] == 'ToyConvV0':
		encoder = ToyConvEncoderV0()

	if decoder_config['model_name'] == 'ToyConv':
		decoder = ToyConvDecoder(**decoder_config['model_params'])
	elif decoder_config['model_name'] == 'Conv':
		decoder = ConvDecoder(**decoder_config['model_params'])
	elif decoder_config['model_name'] == 'ToyConvV0':
		decoder = ToyConvDecoderV0()

	autoencoder = Autoencoder(encoder, 
							  decoder,
							  **autoencoder_config['model_params'])
	autoencoder.apply(weights_init)

	optimizer = torch.optim.Adam(autoencoder.parameters(), 
								 lr=learning_params['learning_rate'])
	scheduler = trainer_params['scheduler']
	if scheduler is not None:
		if scheduler['type'] == 'plateau':
			scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', **scheduler['kwargs'])
		elif scheduler['type'] == 'step':
			scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler['kwargs'])
		elif scheduler['type'] == 'multi_step':
			scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, **scheduler['kwargs'])
        
	stft = STFT(kernel_size=encoder_config['model_params']['nfft'],
				stride=encoder_config['model_params']['hop'],
				coords='polar',
				dB=False)
    
	if autoencoder.vae:
		recon_loss_fx = lambda x, y: vae_perceptual_loss(*x, stft=stft)
		loss_func = {'Perceptual_Loss': recon_loss_fx}
		kld_loss_fx = lambda x, y: vae_kld_loss(*x)
		trainer_params['params']['kld_loss'] = kld_loss_fx
		assert len(trainer_params['params']['weights']) == 2
        
		metric_fx = vae_si_sdr
		metric_func = {'SI_SDR': metric_fx}
	elif autoencoder.vq_vae:
		recon_loss_fx = lambda x, y: vae_perceptual_loss(*x, stft=stft)
		loss_func = {'Perceptual_Loss': recon_loss_fx}
		latent_loss_fx = lambda x, y: vq_vae_latent_loss(*x)
		trainer_params['params']['latent_loss'] = latent_loss_fx
        
		metric_fx = vae_si_sdr
		metric_func = {'SI_SDR': metric_fx}
	else:    
		loss_fx = lambda x, y: perceptual_loss(x, y, stft=stft)
		loss_func = {'Perceptual_Loss': loss_fx}

		metric_fx = si_sdr
		metric_func = {'SI_SDR': metric_fx}
        
	try:
		logistic_fx = lambda x: logistic(x, **trainer_params['weights_func']['kwargs'])
		index = trainer_params['weights_func']['index']
		def weights_func(weights, epoch):
			weights[index] = logistic_fx(epoch)
			return weights
		trainer_params['params']['weights_func'] = weights_func
	except KeyError:
		pass

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

