import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from math import ceil

class BaseEncoder(nn.Module):
	def __init__(self, in_size=(None, 1, 16000),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_down=1,
				 latent_dim=64,
				 clamp=True,
				 vae=False):
		super(BaseEncoder, self).__init__()
		self.in_size = in_size
		self.nfft = nfft
		if hop is None:
			self.hop = self.nfft // 4
		else:
			self.hop = hop
		self.hidden_units = hidden_units
		self.n_res_blocks = n_res_blocks
		self.n_down = n_down
		self.latent_dim = latent_dim
		self.clamp = clamp
		self.vae = vae
		
		self.rep = nn.Conv1d(in_channels=1,
							 out_channels=self.nfft // 2 + 1,
							 kernel_size=self.nfft,
							 stride=self.hop,
							 padding=self.nfft // 2)
		if self.clamp:
			self.rep_relu = ClampedReLU().apply
		else:
			self.rep_relu = nn.ReLU()

		self._rep_size = (nfft // 2 + 1, self._conv1d_out_size(in_size=self.in_size[-1],
						  k=self.nfft,
						  s=self.hop,
						  p=self.nfft // 2))
		self._rep_padding = (ceil(self._rep_size[-1] / 2**n_down) * 2**n_down) - self._rep_size[-1]
		self.rep_padding = Padding1D(pad=self._rep_padding)
		self.conv1 = nn.Conv1d(in_channels=self.nfft // 2 + 1,
							   out_channels=self.hidden_units,
							   kernel_size=3,
							   stride=1,
							   padding=1)
		self.bn1 = nn.BatchNorm1d(self.hidden_units)
		self.down_blocks = nn.ModuleList([self._build_down_blocks(self.n_res_blocks,
																  self.hidden_units) for _ in range(self.n_down)])
		if self.vae:
			_conv_linear_channels = self.latent_dim * 2
		else:
			_conv_linear_channels = self.latent_dim

		self.conv_linear = nn.Conv1d(in_channels=self.hidden_units,
									 out_channels=_conv_linear_channels,
									 kernel_size=1)
		self._latent_size = self._get_latent_size()

	def forward(self, x):
		#print('Input:', x.size())
		rep = self.rep_relu(self.rep(x))
		#print('Representation:', rep.size())
		padded_rep = self.rep_padding(rep)
		#print('Padded Representation:', padded_rep.size())
		x = F.leaky_relu(self.bn1(self.conv1(padded_rep)))
		#print('First Conv:', x.size())
		for db in self.down_blocks:
			x = db(x)
			#print('Down Block:', x.size())
		x = self.conv_linear(x)
		#print('Latent x:', x.size())
		return x

	@staticmethod
	def _conv1d_out_size(in_size, k, s, p):
		return (in_size - (k-1) + 2*p) // s + 1

	def _get_latent_size(self):
		shape = (1, *self.in_size[-2:])
		x = torch.randn(shape)
		x = self.forward(x)
		return x.size()

	@staticmethod
	def _build_down_blocks(n_res, hidden_units, batch_norm=True):
		pre_res_blocks = [ResidualBlock1d(hidden_units, hidden_units) for _ in range(n_res)]
		post_res_blocks = [ResidualBlock1d(hidden_units, hidden_units) for _ in range(n_res)]

		layers = [*pre_res_blocks,
				  nn.Conv1d(in_channels=hidden_units,
							out_channels=hidden_units,
							kernel_size=4,
							stride=2,
							padding=1),
				  nn.LeakyReLU(inplace=True),
				  *post_res_blocks]
		if batch_norm:
			layers.insert(n_res+1, nn.BatchNorm1d(hidden_units))

		down_block = nn.Sequential(*layers)
		return down_block

	@staticmethod
	def _verbose(show, *args):
		if show:
			print(*args)

class BaseDecoder(nn.Module):
	def __init__(self, in_size=(None, 64, 51),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_up=1,
				 latent_dim=64,
				 crop=1,
				 out_size=(None, 1, 16000),
				 clamp=True):
		super(BaseDecoder, self).__init__()
		self.in_size = in_size
		self.nfft = nfft
		if hop is None:
			self.hop = self.nfft // 4
		else:
			self.hop = hop
		self.hidden_units = hidden_units
		self.n_res_blocks = n_res_blocks
		self.n_up = n_up
		self.latent_dim = latent_dim
		self.crop = crop
		self.out_size = out_size
		self.clamp = clamp

		self.conv_linear = nn.Conv1d(in_channels=self.latent_dim,
									 out_channels=self.hidden_units,
									 kernel_size=1)

		self.up_blocks = nn.ModuleList([self._build_up_blocks(self.n_res_blocks,
															  self.hidden_units) for _ in range(self.n_up)])
		self.cropping = Cropping1D(crop)
		self.conv1 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=self.nfft // 2 + 1,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn1 = nn.BatchNorm1d(self.nfft // 2 + 1)

		if self.clamp:
			self.rep_relu = ClampedReLU().apply
		else:
			self.rep_relu = nn.ReLU()
		self.conv_irep = nn.ConvTranspose1d(in_channels=self.nfft // 2 + 1,
											out_channels=1,
											kernel_size=self.nfft,
											stride=self.hop,
											padding=self.nfft // 2)
		self._convtranspose_out_size = self._conv1dtranspose_out_size(in_size=self.in_size[-1] * 2**self.n_up - self.crop,
																	  k=self.nfft,
																	  s=self.hop,
																	  pad=self.nfft // 2)
		self.padding = Padding1D(self.out_size[-1] - self._convtranspose_out_size)
		self.activation = torch.tanh

	def forward(self, x):
		#print('Input:', x.size())
		x = self.conv_linear(x)
		#print('Conv Linear:', x.size())
		for ub in self.up_blocks:
			x = ub(x)
			#print('Up Block:',x.size()) 
		x = F.leaky_relu(self.bn1(self.conv1(x)))
		#print('Conv1:', x.size())
		x = self.rep_relu(self.cropping(x))
		#print('Cropped:', x.size())
		irep = self.conv_irep(x)
		#print('iRep:', irep.size())
		x = self.padding(irep)
		#print('x:', x.size())
		x = self.activation(x)
		return x

	@staticmethod
	def _build_up_blocks(n_res, hidden_units, batch_norm=True):
		pre_res_blocks = [ResidualBlock1d(hidden_units, hidden_units) for _ in range(n_res)]
		post_res_blocks = [ResidualBlock1d(hidden_units, hidden_units) for _ in range(n_res)]

		layers = [*pre_res_blocks,
				  nn.ConvTranspose1d(in_channels=hidden_units,
									 out_channels=hidden_units,
									 kernel_size=4,
									 stride=2,
									 padding=1),
				  nn.LeakyReLU(inplace=True),
				  *post_res_blocks]
		if batch_norm:
			layers.insert(n_res+1, nn.BatchNorm1d(hidden_units))

		up_block = nn.Sequential(*layers)
		return up_block	

	@staticmethod
	def _conv1dtranspose_out_size(in_size, k, s, pad, opad=0):
		out = (in_size-1)*s + k-2*pad + opad
		return out

class BaseAutoencoder(nn.Module):
	def __init__(self, encoder, 
				 decoder, 
				 irmae=False,
				 vae=False,
				 vq_vae=False, 
				 **kwargs):
		super(BaseAutoencoder, self).__init__()
		self.encoder = encoder
		self.decoder = decoder
		self.irmae = irmae
		self.vae = vae
		self.vq_vae = vq_vae

		assert self.vae == encoder.vae, 'Encoder must be set to vae=True'

		self.latent_dim = self.decoder.latent_dim
		if self.irmae:
			self.mlp = MLP(vae=self.vae, **kwargs['mlp_params'])
			assert self.vae == self.mlp.vae, 'MLP must be set to vae=True'

		if self.vq_vae:
			self.vq_embedding = VQEmbeddingEMA()

	def forward(self, x):
		z = self.encoder(x)

		if self.irmae:
			z = self.mlp(z)
		if self.vae:
			mu = z[:, :self.latent_dim]
			logvar = z[:, self.latent_dim:]
			z = self.reparametrization(mu, logvar)
		elif self.vq_vae:
			z, loss, perplexity = self.vq_embedding(z)
		x = self.decoder(z)
		return x

	@staticmethod
	def reparametrization(mu, logvar):
		sigma = torch.exp(0.5 * logvar)
		epsilon = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size())
		return mu + epsilon * sigma