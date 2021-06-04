import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from math import ceil
from Base import BaseModel

class LightweightConvDecoder(BaseModel):
	def __init__(self, in_size=(None, 64, 51),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_up=1,
				 latent_dim=64,
				 crop=1,
				 out_size=(None, 1, 16000),
				 clamp=True,
				 verbose=False):
		super(LightweightConvDecoder, self).__init__()
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
		self.verbose = verbose

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
		self._verbose(self.verbose, 'Input:', x.size())
		x = self.conv_linear(x)
		self._verbose(self.verbose, 'Conv Linear:', x.size())
		for ub in self.up_blocks:
			x = ub(x)
			self._verbose(self.verbose, 'Up Block:',x.size()) 
		x = F.leaky_relu(self.bn1(self.conv1(x)))
		self._verbose(self.verbose, 'Conv1:', x.size())
		x = self.rep_relu(self.cropping(x))
		self._verbose(self.verbose, 'Cropped:', x.size())
		irep = self.conv_irep(x)
		self._verbose(self.verbose, 'iRep:', irep.size())
		x = self.padding(irep)
		self._verbose(self.verbose, 'x:', x.size(), '\n')
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
    
class HeavyDecoder(BaseModel):
	def __init__(self, encoder):
		super(HeavyDecoder, self).__init__()
		self.in_size = encoder._get_latent_size()
		self.nfft = encoder.nfft
		self.hop = encoder.hop
		self.hidden_units = encoder.hidden_units
		self.latent_dim = encoder.latent_dim
		self.res_blocks = encoder.res_blocks

		self.conv_initial = nn.Conv1d(self.latent_dim, self.hidden_units, 1)
		self.blocks = nn.ModuleList([ResidualBlock1d(self.hidden_units, 
													 self.hidden_units)] * self.res_blocks)
		self.conv1 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=self.hidden_units,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn1 = nn.BatchNorm1d(self.hidden_units)
		self.conv2 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=self.hidden_units,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn2 = nn.BatchNorm1d(self.hidden_units)
		self.conv3 = nn.ConvTranspose1d(in_channels=self.hidden_units,
							   out_channels=self.hidden_units,
							   kernel_size=4,
							   stride=2,
							   padding=1)
		self.bn3 = nn.BatchNorm1d(self.hidden_units)
		self.conv4 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=self.hidden_units,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn4 = nn.BatchNorm1d(self.hidden_units)
		self.conv5 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=self.nfft // 2 + 1,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn5 = nn.BatchNorm1d(self.nfft // 2 + 1)
		self.cropping = Cropping1D(encoder.padding.pad)
		self.irep = nn.ConvTranspose1d(in_channels=self.nfft // 2 + 1,
							 out_channels=1,
							 kernel_size=self.nfft,
							 stride=self.hop,
							 padding=self.nfft // 2)
		self._conv1dtranspose_out_size = self._conv1dtranspose_out_size(in_size=encoder._rep_size[-1],
																   k=self.nfft,
																   s=self.hop,
																   pad=self.nfft // 2)
		self.wf_padding = Padding1D(encoder.in_size[-1] - self._conv1dtranspose_out_size)

	def forward(self, x):
		#print(x.size())
		x = self.conv_initial(x)
		#print(x.size())
		for b in self.blocks:
			x = b(x)
		#print(x.size())
		x = F.leaky_relu(self.bn1(self.conv1(x))) + x
		#print(x.size())
		x = F.leaky_relu(self.bn2(self.conv2(x))) + x
		#print(x.size())
		x = F.leaky_relu(self.bn3(self.conv3(x)))
		#print(x.size())
		x = F.leaky_relu(self.bn4(self.conv4(x))) + x
		#print(x.size())
		x = F.leaky_relu(self.bn5(self.conv5(x)))
		#print(x.size())
		x = self.cropping(x)
		#print(x.size())
		x = self.irep(x)
		#print(x.size())
		x = self.wf_padding(x)
		return x
	@staticmethod
	def _conv1dtranspose_out_size(in_size, k, s, pad, opad=0):
		out = (in_size-1)*s + k-2*pad + opad
		return out