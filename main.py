from torchmetrics.classification import MultilabelF1Score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import models
from torchvision.models import ResNet18_Weights
from torch.utils.data import Dataset, DataLoader, random_split


import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import os
import re
import cv2
import pandas
import numpy as np

from nltk import wordpunct_tokenize

BATCH_SIZE = 16
hidden_size = 128

def tokenize(text):

def create_vocab():

class customDataset(Dataset):
    def __init__(self, data):
    def __getitem__(self, index):
    def __len__(self):

class deprecated_model(nn.Module):
    def __init__(self):
    def forward(self, title_tensor, img_tensor):

class BaseModel(nn.Module):
    def __init__(self):
    def forward(self, title_tensor, img_tensor):


def percent_classes(model, title_tensor, img_tensor, topk=3):

def apk(actual : list | int, predicted: list, k: int):

def mapk(actual: list | int, predicted: list, k: int):
    