import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from math import ceil
from Base import BaseModel

class ToyConvDecoder(BaseModel):
	def __init__(self, in_size=(None, 64, 51),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_up=1,
				 latent_dim=64,
				 crop=1,
				 out_size=(None, 1, 16000),
				 transform='conv1d',
				 double_channels = False,  
				 clamp=True,
				 verbose=False,
				 **kwargs):
		super(ToyConvDecoder, self).__init__()
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
		self.transform = transform
		self.double_channels = double_channels

		self.conv_linear = nn.Conv1d(in_channels=self.latent_dim,
									 out_channels=self.hidden_units,
									 kernel_size=1)

		self.up_blocks = nn.ModuleList([self._build_up_blocks(self.n_res_blocks,
															  self.hidden_units) for _ in range(self.n_up)])
		self.cropping = Cropping1D(crop)
		_in_channels = self.nfft // 2 + 1
		if self.transform == 'istft':
			self.coords = kwargs['istft_coords']
			if self.coords == 'polar':
				self.phase_unaware = kwargs['phase_unaware']
				if not self.phase_unaware:
					_in_channels = 2 * _in_channels
			elif self.coords == 'cartesian':
				_in_channels = 2 * _in_channels
		if self.double_channels: _in_channels *= 2
		self.conv1 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=_in_channels,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn1 = nn.BatchNorm1d(_in_channels)

		if self.transform == 'conv1d':
			if self.clamp:
				self.rep_relu = ClampedReLU().apply
			else:
				self.rep_relu = nn.ReLU()
			self.conv_irep = nn.ConvTranspose1d(in_channels=_in_channels,
												out_channels=1,
												kernel_size=self.nfft,
												stride=self.hop,
												padding=self.nfft // 2)
		elif self.transform == 'istft':
			self.irep = iSTFT(kernel_size=self.nfft,
							  stride=self.hop,
							  coords=self.coords)
			self.pre_process = lambda x: torch.chunk(x, dim=1, chunks=2)
			if self.coords == 'polar' and self.phase_unaware:
				self.pre_process = lambda x: x   
		self._convtranspose_out_size = self._conv1dtranspose_out_size(in_size=self.in_size[-1] * 2**self.n_up - self.crop,
																	  k=self.nfft,
																	  s=self.hop,
																	  pad=self.nfft // 2)
		self.padding = Padding1D(self.out_size[-1] - self._convtranspose_out_size)
		self.activation = torch.tanh

	def forward(self, x, *args):
		self._verbose(self.verbose, 'Input:', x.size())
		x = self.conv_linear(x)
		self._verbose(self.verbose, 'Conv Linear:', x.size())
		for ub in self.up_blocks:
			x = ub(x)
			self._verbose(self.verbose, 'Up Block:',x.size()) 
		x = F.leaky_relu(self.bn1(self.conv1(x)))
		self._verbose(self.verbose, 'Conv1:', x.size())
		if self.transform == 'conv1d':
			x = self.rep_relu(self.cropping(x))
			self._verbose(self.verbose, 'Cropped:', x.size())
			irep = self.conv_irep(x)
		elif self.transform == 'istft':
			x = self.cropping(x)
			self._verbose(self.verbose, 'Cropped:', x.size())
			x = self.pre_process(x)
			if self.coords == 'polar' and self.phase_unaware:
				phase = args[0]     
				irep = self.irep(phase, x)
			else:
				irep = self.irep(*x)
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

class ConvDecoder(BaseModel):
	def __init__(self, in_size=(None, 64, 51),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=768,
				 res_blocks=4,
				 latent_dim=64,
				 crop=1,
				 out_size=(None, 1, 16000),
				 transform='conv1d',
				 double_channels=False,
				 clamp=True,
				 verbose=False,
				 **kwargs):
		super(ConvDecoder, self).__init__()
		self.in_size = in_size
		self.nfft = nfft
		self.hop = hop
		self.hidden_units = hidden_units
		self.latent_dim = latent_dim
		self.res_blocks = res_blocks
		self.clamp = clamp
		self.verbose = verbose
		self.out_size = out_size
		self.crop = crop
		self.transform = transform
		self.double_channels = double_channels

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
        
		self.cropping = Cropping1D(self.crop)
		_in_channels = self.nfft // 2 + 1
		if self.transform == 'istft':
			self.coords = kwargs['istft_coords']
			if self.coords == 'polar':
				self.phase_unaware = kwargs['phase_unaware']
				if not self.phase_unaware:
					_in_channels = 2 * _in_channels
			elif self.coords == 'cartesian':
				_in_channels = 2 * _in_channels
		if self.double_channels: _in_channels *= 2
		self.conv5 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=_in_channels,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn5 = nn.BatchNorm1d(_in_channels)

		if self.transform == 'conv1d':
			if self.clamp:
				self.rep_relu = ClampedReLU().apply
			else:
				self.rep_relu = nn.ReLU()
			self.conv_irep = nn.ConvTranspose1d(in_channels=_in_channels,
												out_channels=1,
												kernel_size=self.nfft,
												stride=self.hop,
												padding=self.nfft // 2)
		elif self.transform == 'istft':
			self.irep = iSTFT(kernel_size=self.nfft,
							  stride=self.hop,
							  coords=self.coords)
			self.pre_process = lambda x: torch.chunk(x, dim=1, chunks=2)
			if self.coords == 'polar' and self.phase_unaware:
				self.pre_process = lambda x: x  
                
		self._conv1dtranspose_out_size = self._conv1dtranspose_out_size(in_size=self.in_size[-1] * 2 - self.crop,
																   k=self.nfft,
																   s=self.hop,
																   pad=self.nfft // 2)
		self.wf_padding = Padding1D(self.out_size[-1] - self._conv1dtranspose_out_size)

	def forward(self, x, *args):
		self._verbose(self.verbose, 'Input:', x.size())
		x = self.conv_initial(x)
		self._verbose(self.verbose, 'Conv Initial:', x.size())
		for b in self.blocks:
			x = b(x)
		self._verbose(self.verbose, 'X:', x.size())
		x = F.leaky_relu(self.bn1(self.conv1(x))) + x
		self._verbose(self.verbose, 'X:', x.size())
		x = F.leaky_relu(self.bn2(self.conv2(x))) + x
		self._verbose(self.verbose, 'X:', x.size())
		x = F.leaky_relu(self.bn3(self.conv3(x)))
		self._verbose(self.verbose, 'X:', x.size())
		x = F.leaky_relu(self.bn4(self.conv4(x))) + x
		self._verbose(self.verbose, 'X:', x.size())
		x = F.leaky_relu(self.bn5(self.conv5(x)))
		self._verbose(self.verbose, 'X:', x.size())
		if self.transform == 'conv1d':
			x = self.rep_relu(self.cropping(x))
			self._verbose(self.verbose, 'Cropped:', x.size())
			irep = self.conv_irep(x)
		elif self.transform == 'istft':
			x = self.cropping(x)
			self._verbose(self.verbose, 'Cropped:', x.size())
			x = self.pre_process(x)
			if self.coords == 'polar' and self.phase_unaware:
				phase = args[0]     
				irep = self.irep(phase, x)
			else:
				irep = self.irep(*x)
		self._verbose(self.verbose, 'iRep:', irep.size())
		x = self.wf_padding(irep)
		self._verbose(self.verbose, 'X_hat:', x.size())
		return x

	@staticmethod
	def _conv1dtranspose_out_size(in_size, k, s, pad, opad=0):
		out = (in_size-1)*s + k-2*pad + opad
		return out
    
class ToyConvDecoderV0(BaseModel):
	def __init__(self, in_size=(None, 64, 51),
				 nfft=int(25/1000 * 24414),
				 hop=int(10/1000 * 24414),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_up=1,
				 latent_dim=64,
				 crop=1,
				 out_size=(None, 1, 24414),
				 transform='iSTFT',
				 clamp=True,
				 verbose=False):
		super(ToyConvDecoderV0, self).__init__()
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
		self.transform = transform

		self.conv_linear = nn.Conv1d(in_channels=self.latent_dim,
									 out_channels=self.hidden_units,
									 kernel_size=1)

		self.up_blocks = nn.ModuleList([self._build_up_blocks(self.n_res_blocks,
															  self.hidden_units) for _ in range(self.n_up)])
		self.cropping = Cropping1D(crop)
		if self.transform == 'Conv1d':
			conv1_channels = self.nfft // 2 + 1
		elif self.transform == 'iSTFT':
			conv1_channels = 2*(self.nfft // 2 + 1)
		self.conv1 = nn.ConvTranspose1d(in_channels=self.hidden_units,
										out_channels=conv1_channels,
										kernel_size=3,
										stride=1,
										padding=1)
		self.bn1 = nn.BatchNorm1d(conv1_channels)

		if self.transform == 'Conv1d':
			if self.clamp:
				self.rep_relu = ClampedReLU().apply
			else:
				self.rep_relu = nn.ReLU()
			self.conv_irep = nn.ConvTranspose1d(in_channels=self.nfft // 2 + 1,
												out_channels=1,
												kernel_size=self.nfft,
												stride=self.hop,
												padding=self.nfft // 2)
		elif self.transform == 'iSTFT':
			self.irep = iSTFT(kernel_size=self.nfft,
							  stride=self.hop, coords='cartesian')
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
		if self.transform == 'Conv1d':
			x = self.rep_relu(self.cropping(x))
			self._verbose(self.verbose, 'Cropped:', x.size())
			irep = self.conv_irep(x)
		elif self.transform == 'iSTFT':
			x = self.cropping(x)
			self._verbose(self.verbose, 'Cropped:', x.size())
			real, imag = torch.chunk(x, 
								   dim=1, 
								   chunks=2)
			irep = self.irep(real, imag)
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