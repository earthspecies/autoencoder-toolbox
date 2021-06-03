import torch
import torch.nn as nn
import torch.nn.functional as F
from itertools import combinations
import numpy as np
import tqdm

class HighPassFilter(nn.Module):
	def __init__(self, cutoff_freq, sample_rate, b=0.08, eps=1e-20):
		super(HighPassFilter, self).__init__()
		self.fc = cutoff_freq / sample_rate
		self.b = b
		
		N = int(np.ceil((4 / b)))
		if not N % 2:
			N+=1
		self.N = N
		
		self.epsilon = nn.Parameter(torch.tensor(eps), requires_grad=False)
		self.window = nn.Parameter(torch.blackman_window(N), requires_grad=False)
		
		n = torch.arange(N)
		self.sinc_fx = nn.Parameter(self.sinc(2 * self.fc * (n - (self.N-1) / 2.)), requires_grad=False)
		
	def forward(self, x):
		x = x.view(x.size(0), 1, x.size(-1))
		sinc_fx = self.sinc_fx * self.window
		sinc_fx = torch.true_divide(sinc_fx, torch.sum(sinc_fx))
		sinc_fx = -sinc_fx
		sinc_fx[int((self.N - 1) / 2)] += 1
		output = torch.nn.functional.conv1d(x, sinc_fx.view(-1, 1, self.N), padding=self.N//2)
		return output
		
	def sinc(self, x):
		y = np.pi*torch.where(x==0, self.epsilon, x)
		return torch.true_divide(torch.sin(y), y)  

	def get_config(self):
		config  = {
			'name': 'HighPassFilter',
			'cutoff_freq': self.cutoff_freq,
			'sample_rate': self.sample_rate,
			'b':self.b
		}
		return config

class STFT(nn.Module):
	def __init__(self, 
				 kernel_size, 
				 stride, 
				 coords='polar',
				 dB=False, 
				 epsilon=1e-8):
		super(STFT, self).__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.register_buffer("window", torch.hann_window(self.kernel_size))
		self.coords = coords
		self.epsilon = epsilon
		self.dB = dB

		if self.dB:
			assert self.coords=='polar', 'dB requires magnitude spectrogram'

	def forward(self, x):
		S = torch.stft(x.squeeze(dim=1), 
					   n_fft=self.kernel_size, 
					   hop_length=self.stride, 
					   window=self.window,
					   onesided=True,
					   center='True',
					   pad_mode='reflect',
					   normalized=False)
		S_real = S[:, :, :, 0]
		S_imag = S[:, :, :, 1]
		if self.coords == 'cartesian':
			return S_real, S_imag
		elif self.coords == 'polar':
			S_real = S_real + self.epsilon
			S_imag = S_imag + self.epsilon
			S_phase = torch.atan2(S_imag, S_real)
			S_mag = torch.sqrt(torch.add(torch.pow(S_real, 2), torch.pow(S_imag, 2)))
			if self.dB:
				S_mag = self.amplitude_to_db(S_mag)
			return S_phase, S_mag

	def get_out_size(self, in_size):
		batch, in_filters, L_in = in_size
		L_out = L_in // self.stride + 1
		out_filters = self.kernel_size // 2 + 1
		return (batch, out_filters, L_out)

	def get_config(self):
		config = {
			'name': 'STFT',
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dB scaling': self.dB
		}
		return config

	@staticmethod
	def amplitude_to_db(S, amin=1e-10):
		S = S + amin
		D = torch.mul(torch.log10(S), 20)
		return D

class iSTFT(nn.Module):
	def __init__(self, 
				 kernel_size, 
				 stride, 
				 coords='polar',
				 dB=False):
		super(iSTFT, self).__init__()
		self.kernel_size = kernel_size
		self.stride = stride
		self.register_buffer("window", torch.hann_window(self.kernel_size))
		self.coords = coords
		self.dB = dB

		if self.dB:
			assert self.coords=='polar', 'dB requires magnitude spectrogram'

	def forward(self, S1, S2):
		if self.coords == 'cartesian':
			S_real, S_imag = S1, S2
		elif self.coords == 'polar':
			S_phase, S_mag = S1, S2
			if self.dB:
				S_mag = self.db_to_amplitude(S_mag)
			S_real = torch.mul(S_mag, torch.cos(S_phase)).unsqueeze(dim=-1)
			S_imag = torch.mul(S_mag, torch.sin(S_phase)).unsqueeze(dim=-1)
		S = torch.cat([S_real, S_imag], dim=-1)

		x = torch.istft(S, 
						n_fft=self.kernel_size, 
						hop_length=self.stride, 
						window=self.window).unsqueeze(dim=1)
		return x

	def get_out_size(self, in_size):
		batch, in_filters, L_in = in_size
		L_out = int(L_in - 1) * self.stride
		return (batch, 1, L_out)

	def get_config(self):
		config = {
			'name': 'iSTFT',
			'kernel_size': self.kernel_size,
			'stride': self.stride,
			'dB scaling': self.dB
		}
		return config

	@staticmethod
	def db_to_amplitude(D, amin=1e-10):
		S = torch.pow(10, torch.true_divide(D, 20)) - amin
		return S


class Padding2D(nn.Module):
	def __init__(self, in_size, x_factor=1, y_factor=1, divisible=True):
		super(Padding2D, self).__init__()
		self.x_factor = x_factor
		self.y_factor = y_factor
		self.divisible = divisible
		
		if self.divisible:
			self.x_pad = self.add_padding(in_size[-1], x_factor)
			self.y_pad = self.add_padding(in_size[-2], y_factor)
		else:
			self.x_pad = x_factor
			self.y_pad = y_factor
		
	def forward(self, x):
		ydim, xdim = x.size()[-2:]

		x = F.pad(x, (0, self.x_pad, 0, self.y_pad, 0, 0))
		return x
	
	@staticmethod
	def add_padding(size, factor):
		pad = int(np.ceil(size / factor) * factor) - size
		return pad

	def get_config(self):
		config = {
			'name': 'Padding2D',
			'x_pad': self.x_pad,
			'y_pad': self.y_pad
		}
		return config

class Padding1D(nn.Module):
	def __init__(self, pad):
		super(Padding1D, self).__init__()
		self.pad = pad
		
	def forward(self, x):
		x = F.pad(x, (0, self.pad))
		return x

	def get_config(self):
		config = {
			'name': 'Padding1D',
			'pad': self.pad
		}
		return config

class Cropping2D(nn.Module):
	def __init__(self, x_crop, y_crop):
		super(Cropping2D, self).__init__()
		self.x_crop = x_crop
		self.y_crop = y_crop
	
	def forward(self, x):
		x = torch.split(x, [x.size(-2) - self.y_crop, self.y_crop], dim=-2)[0]
		x = torch.split(x, [x.size(-1) - self.x_crop, self.x_crop], dim=-1)[0]
		return x

	def get_config(self):
		config = {
			'name': 'Cropping2D',
			'x_crop': self.x_crop,
			'y_crop': self.y_crop
		}
		return config

class Cropping1D(nn.Module):
	def __init__(self, crop):
		super(Cropping1D, self).__init__()
		self.crop = crop
	
	def forward(self, x):
		x = torch.split(x, [x.size(-1) - self.crop, self.crop], dim=-1)[0]
		return x

	def get_config(self):
		config = {
			'name': 'Cropping1D',
			'crop': self.crop
		}
		return config

class MCNN(nn.Module):
	def __init__(self, n_heads=8, **kwargs):
		super(MCNN, self).__init__()
		self.n_heads = n_heads
		self.heads = nn.ModuleList([self.construct_head(layers=kwargs['layers'],
														in_filters=kwargs['in_filters'],
														K=kwargs['K'],
														s=kwargs['s'],
														D=kwargs['D']) for _ in range(n_heads)])
		self.ws = nn.Parameter(torch.ones(n_heads), requires_grad=True)
		self.a = nn.Parameter(torch.ones(1), requires_grad=True)
		self.b = nn.Parameter(torch.ones(1), requires_grad=True)
	
	def forward(self, x):
		x = sum([wi*hi(x) for wi, hi in zip(self.ws, self.heads)])
		x = self.LearnableSoftsign(self.a, self.b, x)
		return x
	
	@staticmethod
	def LearnableSoftsign(a, b, x):
		out = a * x / (1 + torch.abs(b * x))
		return out
	
	@staticmethod
	def construct_head(layers, in_filters, K, s, D):
		P = int(D * (K - 1) / 2)
		c_ins = [2**(layers-i) for i in range(1, layers)]
		c_ins = [in_filters, *c_ins]
		c_outs = [2**(layers-i) for i in range(1, layers+1)]
		layers = []
		for c_i, c_o in zip(c_ins, c_outs):
			layers.append(nn.ConvTranspose1d(c_i, c_o, K, s, padding=P))
			layers.append(nn.ELU())
		layers.append(Cropping1D(1))
		head = nn.Sequential(*layers)
		return head

class ClampedReLU(torch.autograd.Function):
	@staticmethod
	def forward(ctx, input):
		ctx.save_for_backward(input)
		return input.clamp(min=0, max=1)

	@staticmethod
	def backward(ctx, grad_output):
		input, = ctx.saved_tensors
		grad_input = grad_output.clone()
		grad_input[input < 0] = 0
		grad_input[input > 1] = 0
		return grad_input

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
		self.leaky_relu = nn.LeakyReLU(inplace=True)
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
		x = self.leaky_relu(x)
		x = self.conv2(x)
		x = self.bn2(x)
		x += residual
		x = self.leaky_relu(x)
		return x

class MLP(nn.Module):
	def __init__(self, 
				 latent_dim=64, 
				 layers=8,
				 channels_first=True,
				 vae=False):
		super(MLP, self).__init__()
		self.latent_dim = latent_dim
		self.layers = layers
		self.channels_first = channels_first
		self.hidden = nn.ModuleList()
		self.vae = vae

		if self.vae:
			hidden_units = 2 * self.latent_dim
		else:
			hidden_units = self.latent_dim
		for k in range(layers):
			linear_layer = nn.Linear(hidden_units, hidden_units, bias=False)
			self.hidden.append(linear_layer)

	def forward(self, z):
		if self.channels_first:
			z = z.permute(0, 2, 1).contiguous()
		for l in self.hidden:
			z = l(z)
		if self.channels_first:
			z = z.permute(0, 2, 1).contiguous()
		return z

class Jitter(nn.Module):
	def __init__(self, p):
		super().__init__()
		self.p = p
		prob = torch.Tensor([p / 2, 1 - p, p / 2])
		self.register_buffer("prob", prob)

	def forward(self, x):
		if not self.training or self.p == 0:
			return x
		else:
			batch_size, sample_size, channels = x.size()

			dist = Categorical(self.prob)
			index = dist.sample(torch.Size([batch_size, sample_size])) - 1
			index[:, 0].clamp_(0, 1)
			index[:, -1].clamp_(-1, 0)
			index += torch.arange(sample_size, device=x.device)

			x = torch.gather(x, 1, index.unsqueeze(-1).expand(-1, -1, channels))
		return x






class VQEmbeddingEMA(nn.Module):
	def __init__(self, 
				 n_embeddings=512, 
				 embedding_dim=64, 
				 commitment_cost=0.25, 
				 decay=0.999, 
				 epsilon=1e-5):
		super(VQEmbeddingEMA, self).__init__()
		self.commitment_cost = commitment_cost
		self.decay = decay
		self.epsilon = epsilon

		init_bound = 1 / 512
		embedding = torch.Tensor(n_embeddings, embedding_dim)
		embedding.uniform_(-init_bound, init_bound)
		self.register_buffer("embedding", embedding)
		self.register_buffer("ema_count", torch.zeros(n_embeddings))
		self.register_buffer("ema_weight", self.embedding.clone())

	def encode(self, x):
		M, D = self.embedding.size()
		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
								torch.sum(x_flat ** 2, dim=1, keepdim=True),
								x_flat, self.embedding.t(),
								alpha=-2.0, beta=1.0)

		indices = torch.argmin(distances.float(), dim=-1)
		quantized = F.embedding(indices, self.embedding)
		quantized = quantized.view_as(x)
		return quantized, indices

	def forward(self, x):
		x = x.permute(0, 2, 1).contiguous()
		print(x.size())
		M, D = self.embedding.size()
		x_flat = x.detach().reshape(-1, D)

		distances = torch.addmm(torch.sum(self.embedding ** 2, dim=1) +
								torch.sum(x_flat ** 2, dim=1, keepdim=True),
								x_flat, self.embedding.t(),
								alpha=-2.0, beta=1.0)

		indices = torch.argmin(distances.float(), dim=-1)
		encodings = F.one_hot(indices, M).float()
		quantized = F.embedding(indices, self.embedding)
		quantized = quantized.view_as(x)

		if self.training:
			self.ema_count = self.decay * self.ema_count + (1 - self.decay) * torch.sum(encodings, dim=0)

			n = torch.sum(self.ema_count)
			self.ema_count = (self.ema_count + self.epsilon) / (n + M * self.epsilon) * n

			dw = torch.matmul(encodings.t(), x_flat)
			self.ema_weight = self.decay * self.ema_weight + (1 - self.decay) * dw

			self.embedding = self.ema_weight / self.ema_count.unsqueeze(-1)

		e_latent_loss = F.mse_loss(x, quantized.detach())
		loss = self.commitment_cost * e_latent_loss

		quantized = x + (quantized - x).detach()

		avg_probs = torch.mean(encodings, dim=0)
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		return quantized.permute(0, 2, 1).contiguous(), loss, perplexity






class VectorQuantizerEMA(nn.Module):
	"""
	Inspired from Sonnet implementation of VQ-VAE https://arxiv.org/abs/1711.00937,
	in https://github.com/deepmind/sonnet/blob/master/sonnet/python/modules/nets/vqvae.py and
	pytorch implementation of it from zalandoresearch in https://github.com/zalandoresearch/pytorch-vq-vae/blob/master/vq-vae.ipynb.
	Implements a slightly modified version of the algorithm presented in
	'Neural Discrete Representation Learning' by van den Oord et al.
	https://arxiv.org/abs/1711.00937
	The difference between VectorQuantizerEMA and VectorQuantizer is that
	this module uses exponential moving averages to update the embedding vectors
	instead of an auxiliary loss. This has the advantage that the embedding
	updates are independent of the choice of optimizer (SGD, RMSProp, Adam, K-Fac,
	...) used for the encoder, decoder and other parts of the architecture. For
	most experiments the EMA version trains faster than the non-EMA version.
	Input any tensor to be quantized. Last dimension will be used as space in
	which to quantize. All other dimensions will be flattened and will be seen
	as different examples to quantize.
	The output tensor will have the same shape as the input.
	For example a tensor with shape [16, 32, 32, 64] will be reshaped into
	[16384, 64] and all 16384 vectors (each of 64 dimensions)  will be quantized
	independently.
	Args:
		embedding_dim: integer representing the dimensionality of the tensors in the
			quantized space. Inputs to the modules must be in this format as well.
		num_embeddings: integer, the number of vectors in the quantized space.
			commitment_cost: scalar which controls the weighting of the loss terms (see
			equation 4 in the paper).
		decay: float, decay for the moving averages.
		epsilon: small float constant to avoid numerical instability.
	"""
	
	def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25, decay=0.999, device='cpu', epsilon=1e-5):
		super(VectorQuantizerEMA, self).__init__()

		self._num_embeddings = num_embeddings
		self._embedding_dim = embedding_dim

		self._embedding = nn.Embedding(self._num_embeddings, self._embedding_dim)
		self._embedding.weight.data.normal_()
		self._commitment_cost = commitment_cost

		self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
		self._ema_w = nn.Parameter(torch.Tensor(num_embeddings, self._embedding_dim))
		self._ema_w.data.normal_()
		
		self._decay = decay
		self._device = device
		self._epsilon = epsilon

	def forward(self, inputs, compute_distances_if_possible=True, record_codebook_stats=False):
		"""
		Connects the module to some inputs.
		Args:
			inputs: Tensor, final dimension must be equal to embedding_dim. All other
				leading dimensions will be flattened and treated as a large batch.
		
		Returns:
			loss: Tensor containing the loss to optimize.
			quantize: Tensor containing the quantized version of the input.
			perplexity: Tensor containing the perplexity of the encodings.
			encodings: Tensor containing the discrete encodings, ie which element
				of the quantized space each input element was mapped to.
			distances
		"""

		# Convert inputs from BCHW -> BHWC
		inputs = inputs.permute(0, 2, 1).contiguous()
		input_shape = inputs.shape
		_, time, batch_size = input_shape
		
		# Flatten input
		flat_input = inputs.view(-1, self._embedding_dim)
		
		# Compute distances between encoded audio frames and embedding vectors
		distances = (torch.sum(flat_input**2, dim=1, keepdim=True) 
					+ torch.sum(self._embedding.weight**2, dim=1)
					- 2 * torch.matmul(flat_input, self._embedding.weight.t()))

		"""
		encoding_indices: Tensor containing the discrete encoding indices, ie
		which element of the quantized space each input element was mapped to.
		"""
		encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
		encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, dtype=torch.float).to(self._device)
		encodings.scatter_(1, encoding_indices, 1)

		# Compute distances between encoding vectors
		if not self.training and compute_distances_if_possible:
			_encoding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(flat_input, r=2)]
			encoding_distances = torch.tensor(_encoding_distances).to(self._device).view(batch_size, -1)
		else:
			encoding_distances = None

		# Compute distances between embedding vectors
		if not self.training and compute_distances_if_possible:
			_embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in combinations(self._embedding.weight, r=2)]
			embedding_distances = torch.tensor(_embedding_distances).to(self._device)
		else:
			embedding_distances = None

		# Sample nearest embedding
		if not self.training and compute_distances_if_possible:
			_frames_vs_embedding_distances = [torch.dist(items[0], items[1], 2).to(self._device) for items in product(flat_input, self._embedding.weight.detach())]
			frames_vs_embedding_distances = torch.tensor(_frames_vs_embedding_distances).to(self._device).view(batch_size, time, -1)
		else:
			frames_vs_embedding_distances = None
		
		# Use EMA to update the embedding vectors
		if self.training:
			self._ema_cluster_size = self._ema_cluster_size * self._decay + \
				(1 - self._decay) * torch.sum(encodings, 0)

			n = torch.sum(self._ema_cluster_size.data)
			self._ema_cluster_size = (
				(self._ema_cluster_size + self._epsilon)
				/ (n + self._num_embeddings * self._epsilon) * n
			)

			dw = torch.matmul(encodings.t(), flat_input)
			self._ema_w = nn.Parameter(self._ema_w * self._decay + (1 - self._decay) * dw)

			self._embedding.weight = nn.Parameter(self._ema_w / self._ema_cluster_size.unsqueeze(1))

		# Quantize and unflatten
		quantized = torch.matmul(encodings, self._embedding.weight).view(input_shape)
		# TODO: Check if the more readable self._embedding.weight.index_select(dim=1, index=encoding_indices) works better

		concatenated_quantized = self._embedding.weight[torch.argmin(distances, dim=1).detach().cpu()] if not self.training or record_codebook_stats else None

		# Loss
		e_latent_loss = torch.mean((quantized.detach() - inputs)**2)
		commitment_loss = self._commitment_cost * e_latent_loss
		vq_loss = commitment_loss

		quantized = inputs + (quantized - inputs).detach()
		avg_probs = torch.mean(encodings, dim=0)

		"""
		The perplexity a useful value to track during training.
		It indicates how many codes are 'active' on average.
		"""
		perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))

		# Convert quantized from BHWC -> BCHW
		return vq_loss, quantized.permute(0, 2, 1).contiguous(), \
			perplexity, encodings.view(batch_size, time, -1), \
			distances.view(batch_size, time, -1), encoding_indices, \
			{'vq_loss': vq_loss.item()}, encoding_distances, embedding_distances, \
			frames_vs_embedding_distances, concatenated_quantized

	@property
	def embedding(self):
		return self._embedding
