import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from math import ceil
from Base import BaseModel

class LightweightConvEncoder(BaseModel):
	def __init__(self, 
				 in_size=(None, 1, 16000),
				 nfft=int(25/1000 * 16000),
				 hop=int(10/1000 * 16000),
				 hidden_units=128,
				 n_res_blocks=1,
				 n_down=1,
				 latent_dim=64,
				 clamp=True,
				 vae=False,
				 verbose=False):
		super(LightweightConvEncoder, self).__init__()
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
		self._verbose(self.verbose, 'Input:', x.size())
		rep = self.rep_relu(self.rep(x))
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
		return x

	@staticmethod
	def _conv1d_out_size(in_size, k, s, p):
		return (in_size - (k-1) + 2*p) // s + 1

	def _get_latent_size(self):
		shape = (1, *self.in_size[-2:])
		x = torch.randn(shape)
		x = self.forward(x)
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



class ResidualBlock1d(nn.Module):
	def __init__(self, in_channels, out_channels, stride=1):
		super(ResidualBlock1d, self).__init__()
		self.conv1 = nn.Conv1d(in_channels=in_channels,
							   out_channels=out_channels,
							   kernel_size=3,
							   stride=stride,
							   padding=1,
							   bias=False)
		self.bn1 = nn.BatchNorm1d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv1d(in_channels=out_channels,
							   out_channels=out_channels,
							   kernel_size=3,
							   stride=stride,
							   padding=1,
							   bias=False)
		self.bn2 = nn.BatchNorm1d(out_channels)

	def forward(self, x):
		residual = x.clone()
		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x += residual
		x = self.relu(x)
		return x

class HeavyEncoder(BaseModel):
	def __init__(self, in_size=(None, 1, 16000), nfft=int(25 / 1000 * 16000), hop=int(10 / 1000 * 16000), hidden_units=768, latent_dim=64, res_blocks=4):
		super(HeavyEncoder, self).__init__()
		self.in_size = in_size
		self.nfft = nfft
		if hop is None:
			self.hop = nfft // 4
		else:
			self.hop = hop
		self.hidden_units = hidden_units
		self.latent_dim = latent_dim
		self.res_blocks = res_blocks
		self.vae = False

		self.rep = nn.Conv1d(in_channels=1,
							 out_channels=self.nfft // 2 + 1,
							 kernel_size=self.nfft,
							 stride=self.hop,
							 padding=self.nfft // 2)
		self._rep_size = (nfft // 2 + 1, self._conv1d_out_size(in_size=self.in_size[-1],
														  k=self.nfft,
														  s=self.hop,
														  p=self.nfft // 2))
		self.padding = Padding1D(pad=self._rep_size[-1] % 2)
		
		self.conv1 = nn.Conv1d(in_channels=self.nfft // 2 + 1,
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
		self.conv_final = nn.Conv1d(self.hidden_units, self.latent_dim, 1)
		self._latent_size = self._get_latent_size()

	def forward(self, x):
		#print(x.size())
		x_rep = F.relu((self.rep(x)))
		#print(x_rep.size())
		x_rep = self.padding(x_rep)
		#print(x_rep.size())
		x_conv1 = F.leaky_relu(self.bn1(self.conv1(x_rep)))
		#print(x_conv1.size())
		x_conv2 = F.leaky_relu(self.bn2(self.conv2(x_conv1)))
		x = x_conv2 + x_conv1
		#print(x.size())
		x_conv3 = F.leaky_relu(self.bn3(self.conv3(x)))

		x_conv4 = F.leaky_relu(self.bn4(self.conv4(x_conv3)))
		x = x_conv4 + x_conv3
		#print(x.size())
		x_conv5 = F.leaky_relu(self.bn5(self.conv5(x)))
		x = x_conv5 + x
		#print(x.size())
		for b in self.blocks:
			x = b(x)
		#print(x.size())
		x = self.conv_final(x)
		#print(x.size())
		return x

	@staticmethod
	def _conv1d_out_size(in_size, k, s, p):
		return (in_size - (k-1) + 2*p) // s + 1

	def _get_latent_size(self):
		shape = (1, *self.in_size[-2:])
		x = torch.randn(shape)
		x = self.forward(x)
		x = x.squeeze(dim=0)
		return x.size()
