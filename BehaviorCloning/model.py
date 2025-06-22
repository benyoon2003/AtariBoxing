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
    if len(os.listdir(path)) < 1: 
        raise Exception("No files found in " + path + " directory when loading actions!")
    
    if len(os.listdir(path)) == 1:
        for file in os.listdir(path):
            filename = os.fsdecode(file)
            if filename == ".DS_Store":
                raise Exception("No files found in " + path + "directory(excluding .DS_Store file!)")

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
    if len(os.listdir(path)) < 1: 
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
        observation = self.obs.get(idx)

        # Convert to float32 and normalize to [0, 1]
        observation = torch.tensor(observation, dtype=torch.float32) / 255.0

        # Add a channel dimension (C, H, W) expected by Conv2d
        if len(observation.shape) == 2:
            observation = observation.unsqueeze(0)

        action = int(self.actions.iloc[idx, 0])  # make sure it's an int for CrossEntropyLoss

        if self.transform:
            observation = self.transform(observation)
        if self.target_transform:
            action = self.target_transform(action)

        return observation, action
    
dataset = CustomDataset('actions', 'observations')

#hyperparams

learning_rate = 3e-3
batch_size = 64
epochs = 60


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

        self.conv_stack = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=8, stride=4),  # Input: (1, H, W)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # Dummy forward pass to calculate flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 210, 160)  # adjust to your actual input H, W
            dummy_output = self.conv_stack(dummy_input)
            flattened_size = dummy_output.view(1, -1).shape[1]

        self.fc_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, 100),  # Replace 18 with actual number of actions
            nn.ReLU(),
            nn.Linear(100, 18)  # Replace 18 with actual number of actions
        )

    def forward(self, x):
        x = self.conv_stack(x)
        x = self.fc_stack(x)
        return x
    

model = NeuralNetwork()



def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        #pred/loss calculation

        # X = X.to(torch.double)
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

    with open("loss_data", "a") as f:
        f.write(str(test_loss) + ", ")

    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# Want to visualize the model actually working
def evaluate(model_path = "behavioral_cloning.pt", episodes=1):
    env = gym.make("ALE/Boxing-v5", render_mode="human")
    policy_net = NeuralNetwork().to(device)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()
        watch_agent(policy_net, env, episodes)
    else:
        print("No trained model found.")
    env.close()

def select_action(state, policy_net, action_size, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size) # do a random action
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) # convert state to float tensor, batch dimension for GPU
        with torch.no_grad(): 
            state_tensor = state_tensor.permute(3, 0, 1, 2)
            q_values = policy_net(state_tensor) # get q values for each action
        # im not entirely sure why but this is required to get it to run
        q_argmax = -1
        for i in range(0, 3):
            if q_values[i].argmax().item() > q_argmax:
                q_argmax = q_values[i].argmax().item()
        return q_argmax # Get index with max q value and convert it into int

def watch_agent(policy_net, env, episodes=1):
    for episode in range(episodes):
        state, _ = env.reset()
        frames = deque(maxlen=4)
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = select_action(state, policy_net, env.action_space.n, 0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Evaluation Episode {episode}, Total Reward: {total_reward}")



loss_fn = nn.CrossEntropyLoss()


optimizer = torch.optim.ASGD(model.parameters(), lr= learning_rate)

for t in range(epochs):
    print("Epoch:", t)
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done")
with open("loss_data", "a") as f:
    f.write("|||")

torch.save(model.state_dict(), "behavioral_cloning.pt")
evaluate()
