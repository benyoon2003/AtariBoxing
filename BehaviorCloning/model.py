# Source/inspo for majority of implementation: https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html

import argparse
import collections
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
import glob
from collections import deque, namedtuple
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import ale_py
import natsort

def convertActions(act_dir):
    dataframe = pd.DataFrame()
    path = os.getcwd() + "/" + act_dir

    # because the .DS_Store file exists for some reason - i have no idea what it is but if it stops existing this should still throw the exception
    if len(os.listdir(path)) <= 1: 
        raise Exception("No files found in " + path + " directory when loading actions!")

    for file in os.listdir(path):
        filename = os.fsdecode(file)
        if filename.endswith(".npy"):
            np_arr = np.load(path + "/" + filename, allow_pickle=True)
            np_dataframe = pd.DataFrame(np_arr)
            dataframe = pd.concat([dataframe, np_dataframe])

    
    return dataframe

def convertObservations(obs_dir):
    dic = dict()
    path = os.getcwd() + "/" + obs_dir

    # because the .DS_Store file exists for some reason - i have no idea what it is but if it stops existing this should still throw the exception
    if len(os.listdir(path)) <= 1: 
        raise Exception("No files found in " + path + " directory when loading actions!")
    
    number_of_added_files = 0

    for direct in os.listdir(path):
        dir_path = path + "/" + os.fsdecode(direct)
        if os.path.isdir(dir_path):
            for file in natsort.natsorted(os.listdir(dir_path)):
                filename = os.fsdecode(file)
                if filename.endswith(".npy"):
                    index_of_underscore = filename.index("_")
                    index_of_period = filename.index(".")
                    count = filename[index_of_underscore+1:index_of_period]
                    np_arr = np.load(dir_path + "/" + filename, allow_pickle=True)
                    dic.update({number_of_added_files : np_arr})
                    number_of_added_files += 1

    
    return collections.OrderedDict(sorted(dic.items()))

class CustomDataset(Dataset):
    def __init__(self, actions_dir, observations_dir, transform=None, target_transform=None):
        self.actions = convertActions(actions_dir)
        self.obs = convertObservations(observations_dir)
        self.transform = transform
        self.target_transform = target_transform


    def __len__(self):
        return len(self.actions)

    def __getitem__(self, idx):
        observation = torch.from_numpy(self.obs.get(idx))
        observation = observation.to(torch.double)
        action = self.actions.iloc[idx, 0]
        if self.transform:
            observation = self.transform(observation)
        if self.target_transform:
            action = self.target_transform(action)
        return observation, action
    
dataset = CustomDataset('actions', 'observations')

#hyperparams

learning_rate = 1e-3
batch_size = 64
epochs = 3


train_data, test_data = torch.utils.data.random_split(dataset, [0.8, 0.2])

train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

env = gym.make("ALE/Boxing-v5")

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(33600, 33600),
            nn.ReLU(),
            nn.Linear(33600, 3600),
            nn.ReLU(),
            nn.Linear(3600, 100),
            nn.ReLU(),
            nn.Linear(100, 18),
        )
        self.double()
    
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork()



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #pred/loss calculation

        print(y.dtype)

        # X = X.to(torch.double)

        print(X.dtype)

        pred = model(X)
        loss = loss_fn(pred, y)

        #backpropogation
        loss.backward()

        #optimizing with step func
        optimizer.step()

        #reseting with zero_grad
        optimizer.zero_grad()

        if batch % 10 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)

            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr= learning_rate)

for t in range(epochs):
    print("Epoch:", t)
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done")

# actions = convertActions('actions')
# print(actions)
# obs = convertObservations('observations')
# print(obs)
