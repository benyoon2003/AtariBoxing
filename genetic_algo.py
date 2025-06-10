import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import copy
import random
import numpy as np
from collections import deque
import multiprocessing as mp
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ENV_NAME = "ALE/Boxing-v5"
env = gym.make(ENV_NAME, render_mode=None)


class CNNPolicyNet(nn.Module):
    def __init__(self, output_dim):
        super(CNNPolicyNet, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Compute size after conv layers for FC
        # Input 84x84, conv1 -> ((84-8)/4)+1=20, conv2 -> ((20-4)/2)+1=9, conv3 -> ((9-3)/1)+1=7
        # So final feature map is 7x7 with 64 channels
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, output_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)  # flatten
        x = F.relu(self.fc1(x))
        return self.out(x)

# Formats single frame
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) # Make frame grayscale
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA) # Resize to 84x84 
    return frame / 255.0 # Resize and normalize

# Creates state by stacking last 4 frames
def stack_frames(frames, state, is_new_episode):
    frame = preprocess(state)
    if is_new_episode:
        frames = deque([frame for _ in range(4)], maxlen=4) # initialize start state
    else:
        frames.append(frame)
    stacked_state = np.stack(frames, axis=0) # shows motion over last 4 frames

    return frames, stacked_state

