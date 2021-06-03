import torch
import torch.nn as nn
import torch.nn.functional as F

from Layers import *
from Base import BaseModel, BaseAutoencoder
from Encoders import * 
from Decoders import *

class Autoencoder(BaseAutoencoder):
	def __init__(self,
				 encoder,
				 decoder,
				 irmae=False,
				 vae=False,
				 vq_vae=False,
				 **kwargs):
		self.encoder = encoder
		self.decoder = decoder
		self.irmae = irmae
		self.vae = vae
		self.vq_vae = vae

		assert self.vae == encoder.vae, 'Encoder must be set to vae=True'

		self.latent_dim = self.encoder.latent_dim
		if self.irmae:
			self.mlp = MLP(vae=self.vae, **kwargs['mlp_params'])
			assert self.vae == self.mlp.vae, 'MLP must be set to vae=True'

		if self.vq_vae:
			self.vq_embedding = VQEmbeddingEMA()

	def encode(self, x):
		z = self.encoder(x)
		return z

	def latent(self, z):
		if self.irmae:
			z = self.mlp(z)
		if self.vae:
			mu = z[:, :self.latent_dim]
			logvar = z[:, self.latent_dim:]
			z = self.reparametrization(mu, logvar)
			return z, mu, logvar
		elif self.vq_vae:
			z, loss, perplexity = self.vq_embedding(z)
			return z, loss, perplexity
		else:
			return z

	def decode(self, z):
		x = self.decoder(z)
		return x

	def encode_to_latent(self, x):
		z = self.encode(x)

		latents = self.latent(z)
		if self.vae:
			z, mu, logvar = latents
		elif self.vq_vae:
			z, loss, perplexity = latents
		else:
			z = latents
		return z

	def forward(self, x):
		z = self.encode(x)
		
		latents = self.latent(z)
		if self.vae:
			z, mu, logvar = latents
		elif self.vq_vae:
			z, loss, perplexity = latents
		else:
			z = latents

		x_hat = self.decode(z)

		if self.vae:
			return x_hat, x, mu, logvar, z
		elif self.vq_vae:
			return x_hat, x, z, loss, perplexity
		else:
			return x_hat, x

	def sample(self, n_samples, current_device):
		"""TO DO: fix sampling to account for 2D latent rep"""
		z = torch.randn(n_samples, self.latent_dim)
		z = z.to(current_device)
		x_hat = self.decode(z)
		return z

	def generate(self, x):
		return self.forward(x)[0]