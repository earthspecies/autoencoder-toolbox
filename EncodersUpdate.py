import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from math import ceil
from Base import BaseModel

class ToyConvEncoder(BaseModel):
	def __init__(self, 
				 in_size=(None, 1, 16000),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_down=1,
				 latent_dim=64,
				 transform = 'conv1d',
				 double_channels = False,
				 clamp=True,
				 vae=False,
				 verbose=False,
				 **kwargs):
		super(ToyConvEncoder, self).__init__()
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
		self.verbose = verbose
		self.transform = transform
		self.double_channels = double_channels
		
		if self.transform == 'conv1d':
			_out_channels = self.nfft // 2 + 1
			if self.double_channels: _out_channels *= 2
			self.rep = nn.Conv1d(in_channels=1,
								 out_channels=_out_channels,
								 kernel_size=self.nfft,
								 stride=self.hop,
								 padding=self.nfft // 2)
			if self.clamp:
				self.rep_relu = ClampedReLU().apply
			else:
				self.rep_relu = nn.ReLU()
		elif self.transform == 'stft':
			self.post_process = lambda x: torch.cat(list(x), dim=1) 
			_out_channels = 2 * (self.nfft // 2 + 1)
			self.coords = kwargs['stft_coords']
			if self.coords == 'polar':
				self.phase_unaware = kwargs['phase_unaware']
				if self.phase_unaware:
					_out_channels = self.nfft // 2 + 1
					self.post_process = lambda x: list(x)
			self.rep = STFT(kernel_size=self.nfft,
							stride=self.hop,
							coords=self.coords)                 

		self._rep_size = (_out_channels, self._conv1d_out_size(in_size=self.in_size[-1],
						  k=self.nfft,
						  s=self.hop,
						  p=self.nfft // 2))
		self._rep_padding = (ceil(self._rep_size[-1] / 2**n_down) * 2**n_down) - self._rep_size[-1]
		self.rep_padding = Padding1D(pad=self._rep_padding)
		self.conv1 = nn.Conv1d(in_channels=_out_channels,
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
		self._verbose(self.verbose, 'Input:', x.size())

		if self.transform == 'conv1d':
			rep = self.rep_relu(self.rep(x))
		elif self.transform == 'stft':
			rep = self.post_process(self.rep(x))
			if self.coords == 'polar' and self.phase_unaware:
				phase, rep = rep
			else:
				phase = None
		self._verbose(self.verbose, 'Representation:', rep.size())
		padded_rep = self.rep_padding(rep)
		self._verbose(self.verbose, 'Padded Representation:', padded_rep.size())
		x = F.leaky_relu(self.bn1(self.conv1(padded_rep)))
		self._verbose(self.verbose, 'First Conv:', x.size())
		for db in self.down_blocks:
			x = db(x)
			self._verbose(self.verbose, 'Down Block:', x.size())
		x = self.conv_linear(x)
		self._verbose(self.verbose, 'Latent x:', x.size(), '\n')
		
		if self.transform == 'stft' and phase is not None:
			return phase, x
		else:
			return x

	@staticmethod
	def _conv1d_out_size(in_size, k, s, p):
		return (in_size - (k-1) + 2*p) // s + 1

	def _get_latent_size(self):
		shape = (1, *self.in_size[-2:])
		x = torch.randn(shape)
		x = self.forward(x)
		if type(x) == tuple:
			x = x[-1]
		x = x.squeeze(dim=0)
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

class ConvEncoder(BaseModel):
	def __init__(self, 
				 in_size=(None, 1, 16000), 
				 nfft=int(25 / 1000 * 16000), 
				 hop=int(10 / 1000 * 16000), 
				 hidden_units=768, 
				 latent_dim=64, 
				 res_blocks=4,
				 transform='conv1d',
				 clamp=True,
				 double_channels=False,
				 vae=False,
				 verbose=False,
				 **kwargs):
		super(ConvEncoder, self).__init__()
		self.in_size = in_size
		self.nfft = nfft
		if hop is None:
			self.hop = nfft // 4
		else:
			self.hop = hop
		self.hidden_units = hidden_units
		self.latent_dim = latent_dim
		self.res_blocks = res_blocks
		self.vae = vae
		self.clamp = clamp
		self.verbose = verbose
		self.transform = transform
		self.double_channels = double_channels

		if self.transform == 'conv1d':
			_out_channels = self.nfft // 2 + 1
			if self.double_channels: _out_channels *= 2
			self.rep = nn.Conv1d(in_channels=1,
								 out_channels=_out_channels,
								 kernel_size=self.nfft,
								 stride=self.hop,
								 padding=self.nfft // 2)
			if self.clamp:
				self.rep_relu = ClampedReLU().apply
			else:
				self.rep_relu = nn.ReLU()
		elif self.transform == 'stft':
			self.post_process = lambda x: torch.cat(list(x), dim=1) 
			_out_channels = 2 * (self.nfft // 2 + 1)
			self.coords = kwargs['stft_coords']
			if self.coords == 'polar':
				self.phase_unaware = kwargs['phase_unaware']
				if self.phase_unaware:
					_out_channels = self.nfft // 2 + 1
					self.post_process = lambda x: list(x)
			self.rep = STFT(kernel_size=self.nfft,
							stride=self.hop,
							coords=self.coords)                 

		self._rep_size = (_out_channels, self._conv1d_out_size(in_size=self.in_size[-1],
						  k=self.nfft,
						  s=self.hop,
						  p=self.nfft // 2))
        
		self.padding = Padding1D(pad=self._rep_size[-1] % 2)
		
		self.conv1 = nn.Conv1d(in_channels=_out_channels,
							   out_channels=self.hidden_units,
							   kernel_size=3,
							   stride=1,
							   padding=1)
		self.bn1 = nn.BatchNorm1d(self.hidden_units)
		self.conv2 = nn.Conv1d(in_channels=self.hidden_units,
							   out_channels=self.hidden_units,
							   kernel_size=3,
							   stride=1,
							   padding=1)
		self.bn2 = nn.BatchNorm1d(self.hidden_units)

		self.conv3 = nn.Conv1d(in_channels=self.hidden_units,
							   out_channels=self.hidden_units,
							   kernel_size=4,
							   stride=2,
							   padding=1)
		self.bn3 = nn.BatchNorm1d(self.hidden_units)

		self.conv4 = nn.Conv1d(in_channels=self.hidden_units,
							   out_channels=self.hidden_units,
							   kernel_size=3,
							   stride=1,
							   padding=1)
		self.bn4 = nn.BatchNorm1d(self.hidden_units)
		self.conv5 = nn.Conv1d(in_channels=self.hidden_units,
							   out_channels=self.hidden_units,
							   kernel_size=3,
							   stride=1,
							   padding=1)
		self.bn5 = nn.BatchNorm1d(self.hidden_units)

		self.blocks = nn.ModuleList([ResidualBlock1d(self.hidden_units, 
													 self.hidden_units)] * self.res_blocks)
		if self.vae:
			_conv_linear_channels = self.latent_dim * 2
		else:
			_conv_linear_channels = self.latent_dim
		self.conv_final = nn.Conv1d(self.hidden_units, _conv_linear_channels, 1)
		self._latent_size = self._get_latent_size()

	def forward(self, x):
		self._verbose(self.verbose, 'Input:', x.size())

		if self.transform == 'conv1d':
			x_rep = self.rep_relu(self.rep(x))
		elif self.transform == 'stft':
			x_rep = self.post_process(self.rep(x))
			if self.coords == 'polar' and self.phase_unaware:
				phase, x_rep = x_rep
			else:
				phase = None
                
		self._verbose(self.verbose, 'Representation:', x_rep.size())
		x_rep = self.padding(x_rep)
		self._verbose(self.verbose, 'Padded Representation:', x_rep.size())
		x_conv1 = F.leaky_relu(self.bn1(self.conv1(x_rep)))
		self._verbose(self.verbose, 'Conv1:', x_conv1.size())
		x_conv2 = F.leaky_relu(self.bn2(self.conv2(x_conv1)))
		x = x_conv2 + x_conv1
		self._verbose(self.verbose, 'X:', x.size())
		x_conv3 = F.leaky_relu(self.bn3(self.conv3(x)))
		self._verbose(self.verbose, 'x_conv3:', x_conv3.size())
		x_conv4 = F.leaky_relu(self.bn4(self.conv4(x_conv3)))
		x = x_conv4 + x_conv3
		self._verbose(self.verbose, 'X:', x.size())
		x_conv5 = F.leaky_relu(self.bn5(self.conv5(x)))
		x = x_conv5 + x
		self._verbose(self.verbose, 'X:', x.size())
		for b in self.blocks:
			x = b(x)
			self._verbose(self.verbose, 'X:', x.size())
		x = self.conv_final(x)
		self._verbose(self.verbose, 'Latent:', x.size())
        
		if self.transform == 'stft' and phase is not None:
			return phase, x
		else:
			return x

	@staticmethod
	def _conv1d_out_size(in_size, k, s, p):
		return (in_size - (k-1) + 2*p) // s + 1

	def _get_latent_size(self):
		v = self.verbose
		self.verbose = False
		shape = (1, *self.in_size[-2:])
		x = torch.randn(shape)
		x = self.forward(x)
		if type(x) == tuple:
			x = x[-1]
		x = x.squeeze(dim=0)
		self.verbose = v
		return x.size()
