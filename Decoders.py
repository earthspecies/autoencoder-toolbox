import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from math import ceil

class LightweightConvDecoder(nn.Module):
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