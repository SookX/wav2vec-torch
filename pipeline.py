import torch
import torch.nn as nn
from torchaudio.datasets import LIBRISPEECH

from utils import get_device
from model.wav2vec2 import Wav2Vec

train_dataset = LIBRISPEECH("./dataset/", url="train-clean-100", download=False)
val_dataset = LIBRISPEECH("./dataset/", url="dev-clean", download=False)