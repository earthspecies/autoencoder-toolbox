import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
import numpy as np

class BaseModel(nn.Module):
	@abstractmethod
	def forward(self, *input):
		raise NotImplementedError

	def __str__(self):
		model_parameters = filter(lambda p: p.requires_grad, self.parameters())
		params = sum([np.prod(p.size()) for p in model_parameters])
		return super(BaseModel, self).__str__() + f'\nTrainable parameters: {params}'

	@staticmethod
	def _verbose(show, *args):
		if show:
			print(*args)

class BaseAutoencoder(BaseModel):	
	def __init__(self):
		super(BaseAutoencoder, self).__init__()

	def encode(self, *input):
		raise NotImplementedError

	def decode(self, *input):
		raise NotImplementedError

	def sample(self, batch_size, current_device, **kwargs):
		raise RuntimeWarning()

	def generate(self, x, **kwargs):
		raise NotImplementedError

	def latent(self, *input):
		raise NotImplementedError

	@abstractmethod
	def forward(self, *inputs):
		raise NotImplementedError

	@staticmethod
	def reparametrization(mu, logvar):
		sigma = torch.exp(0.5 * logvar)
		epsilon = torch.distributions.normal.Normal(0, 1).sample(sample_shape=sigma.size()).to(mu.device)
		return mu + epsilon * sigma
