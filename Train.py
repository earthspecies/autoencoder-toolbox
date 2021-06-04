import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse
import json

from Datasets import *
from Autoencoder import *
from Encoders import *
from Decoders import *
import PyFireUpdate

from Losses import *
from Metrics import *
import matplotlib.pyplot as plt

if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	plt.plot([1, 2, 3])
	plt.show()
