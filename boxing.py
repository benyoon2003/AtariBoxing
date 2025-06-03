import argparse
import gymnasium as gym
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import os
import cv2

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")

class DQN(nn.Module):
    def __init__(self, input_channels, action_size):
        super(DQN, self).__init__()

        # First CNN reduces spatial size, detects broad patterns
        self.conv1 = nn.Conv2d(input_channels, 32, 8, stride=4)

        # Second CNN Builds on first CNN to detect more complex shapes
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)

        # Third CNN captures fine-grained details like small changes in position, and movement direction
        self.conv3 = nn.Conv2d(64, 64, 3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, action_size)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        return self.out(x)

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).unsqueeze(1).to(device),
            torch.FloatTensor(rewards).unsqueeze(1).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).unsqueeze(1).to(device),
        )

    def __len__(self):
        return len(self.buffer)

def preprocess(frame):
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
    return frame / 255.0

def stack_frames(frames, state, is_new_episode):
    frame = preprocess(state)
    if is_new_episode:
        frames = deque([frame for _ in range(4)], maxlen=4)
    else:
        frames.append(frame)
    stacked_state = np.stack(frames, axis=0)
    return frames, stacked_state

def select_action(state, policy_net, action_size, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size)
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = policy_net(state_tensor)
        return q_values.argmax().item()

def train():
    env = gym.make("ALE/Boxing-v5")
    action_size = env.action_space.n
    policy_net = DQN(4, action_size).to(device)
    target_net = DQN(4, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    memory = ReplayBuffer(3000)

    epsilon = 1.0
    EPSILON_END = 0.1
    EPSILON_DECAY = 0.995
    GAMMA = 0.99
    BATCH_SIZE = 32
    TARGET_UPDATE = 10
    EPISODES = 100
    MODEL_PATH = "models/dqn_policy_net.pth"

    for episode in range(EPISODES):
        state, _ = env.reset()
        frames = deque(maxlen=4)
        frames, stacked_state = stack_frames(frames, state, True)
        total_reward = 0
        done = False

        while not done:
            action = select_action(stacked_state, policy_net, action_size, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames, next_stacked_state = stack_frames(frames, next_state, False)

            memory.push(stacked_state, action, reward, next_stacked_state, done)
            stacked_state = next_stacked_state
            total_reward += reward

            if len(memory) >= BATCH_SIZE:
                states, actions, rewards, next_states, dones = memory.sample(BATCH_SIZE)

                # Compute targets
                with torch.no_grad():
                    next_q = target_net(next_states).max(1, keepdim=True)[0]
                    target_q = rewards + GAMMA * next_q * (1 - dones)

                # Current Q
                current_q = policy_net(states).gather(1, actions)

                # Loss and optimization
                loss = nn.MSELoss()(current_q, target_q)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        epsilon = max(EPSILON_END, epsilon * EPSILON_DECAY)
        if episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode}, Total Reward: {total_reward}, Epsilon: {epsilon:.2f}")

    env.close()
    torch.save(policy_net.state_dict(), MODEL_PATH)
    print("Model saved to", MODEL_PATH)

def evaluate():
    env = gym.make("ALE/Boxing-v5", render_mode="human")
    action_size = env.action_space.n
    policy_net = DQN(4, action_size).to(device)
    MODEL_PATH = "models/dqn_policy_net.pth"
    if os.path.exists(MODEL_PATH):
        policy_net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
        policy_net.eval()
        print("Loaded trained model.")
        watch_agent(policy_net, env, episodes=1)
    else:
        print("No trained model found.")
    env.close()

def watch_agent(policy_net, env, episodes=1):
    for episode in range(episodes):
        state, _ = env.reset()
        frames = deque(maxlen=4)
        frames, stacked_state = stack_frames(frames, state, True)
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = select_action(stacked_state, policy_net, env.action_space.n, 0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames, stacked_state = stack_frames(frames, next_state, False)
            total_reward += reward
        print(f"Evaluation Episode {episode}, Total Reward: {total_reward}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    args = parser.parse_args()

    if args.train:
        train()
    elif args.eval:
        evaluate()
    else:
        print("Specify either --train or --eval.")

if __name__ == "__main__":
    main()
