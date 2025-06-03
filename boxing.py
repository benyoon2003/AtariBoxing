import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if (torch.cuda.is_available()):
    print("Using GPU")

# Preprocessing function
def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)  # Convert to grayscale
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)  # Resize to 84x84
    return frame / 255.0  # Normalize pixel values

# Stack frames
def stack_frames(frames, state, is_new_episode):
    frame = preprocess(state)
    if is_new_episode:
        frames = deque([frame for _ in range(4)], maxlen=4)
    else:
        frames.append(frame)
    stacked_state = np.stack(frames, axis=0)
    return frames, stacked_state

def watch_agent(policy_net, env, episodes=1):
    for episode in range(episodes):
        state, _ = env.reset()
        frames = deque(maxlen=4)
        frames, stacked_state = stack_frames(frames, state, True)
        total_reward = 0
        done = False

        while not done:
            env.render()  # Render the environment
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames, stacked_state = stack_frames(frames, next_state, False)
            total_reward += reward

        print(f"Evaluation Episode {episode}, Total Reward: {total_reward}")
    env.close()

# Define the Q-network
class DQN(nn.Module):
    def __init__(self, action_size):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7*7*64, 512)
        self.out = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))  # [batch, 32, 20, 20]
        x = torch.relu(self.conv2(x))  # [batch, 64, 9, 9]
        x = torch.relu(self.conv3(x))  # [batch, 64, 7, 7]
        x = x.view(x.size(0), -1)      # Flatten
        x = torch.relu(self.fc1(x))
        return self.out(x)

# Hyperparameters
EPISODES = 5
GAMMA = 0.99
LR = 1e-4
BATCH_SIZE = 32
MEMORY_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.995
TARGET_UPDATE = 10

# Initialize environment
env = gym.make("ALE/Boxing-v5", render_mode="human")
action_size = env.action_space.n

# Initialize networks
policy_net = DQN(action_size).to(device)
target_net = DQN(action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LR)
memory = deque(maxlen=MEMORY_SIZE)

epsilon = EPSILON_START

for episode in range(EPISODES):
    print(f"Training episode: {episode}")
    state, _ = env.reset()
    frames = deque(maxlen=4)
    frames, stacked_state = stack_frames(frames, state, True)
    total_reward = 0
    done = False

    while not done:
        if random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.FloatTensor(stacked_state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
            action = q_values.argmax().item()

        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        frames, next_stacked_state = stack_frames(frames, next_state, False)
        memory.append((stacked_state, action, reward, next_stacked_state, done))
        stacked_state = next_stacked_state
        total_reward += reward

        if len(memory) >= BATCH_SIZE:
            batch = random.sample(memory, BATCH_SIZE)
            states, actions, rewards, next_states, dones = zip(*batch)

            states = torch.FloatTensor(np.array(states)).to(device)
            actions = torch.LongTensor(actions).unsqueeze(1).to(device)
            next_states = torch.FloatTensor(np.array(next_states)).to(device)
            rewards = torch.FloatTensor(np.array(rewards)).unsqueeze(1).to(device)
            dones = torch.FloatTensor(np.array(dones)).unsqueeze(1).to(device)

            current_q = policy_net(states).gather(1, actions)
            next_q = target_net(next_states).max(1)[0].unsqueeze(1)
            target_q = rewards + (GAMMA * next_q * (1 - dones))

            loss = nn.MSELoss()(current_q, target_q)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

    print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

env.close()

watch_agent(policy_net, env, episodes=1)
