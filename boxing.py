import argparse
import gymnasium as gym
import numpy as np
import random
from collections import deque, namedtuple
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

env = gym.make("ALE/Boxing-v5") # Atari boxing environment

# Use GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    print("Using GPU")

# A cyclic buffer of bounded size that holds the transitions observed recently, observed significantly worse performance without ReplayMemory
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args)) # transition tuple maps (s, a) to (s*, R)

    # Samples random batch for training. Stabilizes and improves DQN
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# CNN based DQN
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
        x = torch.relu(self.conv1(x)) # Non linearity
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
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

# Select action with Epsilon Greedy policy
def select_action(state, policy_net, action_size, epsilon):
    if random.random() < epsilon:
        return random.randrange(action_size) # do a random action
    else:
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device) # convert state to float tensor, batch dimension for GPU
        with torch.no_grad(): 
            q_values = policy_net(state_tensor) # get q values for each action
        return q_values.argmax().item() # Get index with max q value and convert it into int


def optimize_model(memory, batch_size, target_net, policy_net, gamma, optimizer):
    if len(memory) < batch_size:
        return

    # Sample batch and convert to Transition of batches
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))

    # Create mask of non-terminal states
    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)),
        device=device, dtype=torch.bool
    )

    # Handle next_state. None for terminal states
    non_final_next_states = torch.FloatTensor(
        np.array([s for s in batch.next_state if s is not None])
    ).to(device)

    state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
    action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
    reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)

    # Compute Q(s_t, a) from policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states using target_net
    next_state_values = torch.zeros(batch_size, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values

    # Compute expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch.squeeze(1)

    # Use SmoothL1Loss (Huber)
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    # Backprop
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)  # gradient clipping
    optimizer.step()


def train(batch_size = 32, gamma = 0.99, epsilon = 1.0, epsilon_limit = 0.1, decay_factor = 0.995, LR = 1e-4, model_path = "models/dqn.pth", episodes = 100):   
    action_size = env.action_space.n # get num actions
    state, info = env.reset()

    policy_net = DQN(4, action_size).to(device)
    target_net = DQN(4, action_size).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    optimizer = optim.Adam(policy_net.parameters(), lr=LR, amsgrad=True)

    # Kept this low coz it keeps crashing my laptop otherwise
    memory = ReplayMemory(3000)
    avg_reward = 0

    for episode in range(episodes):
        state, _ = env.reset() # Reset before every episode
        frames = deque(maxlen=4)
        frames, state = stack_frames(frames, state, True) # get initial state
        total_reward = 0
        done = False

        while not done:
            action = select_action(state, policy_net, action_size, epsilon)
            raw_next_state, reward, terminated, truncated, _ = env.step(action) # execute action
            done = terminated or truncated
            frames, processed_next_state = stack_frames(frames, raw_next_state, False) # get deque of last 4 frames

            memory.push(state, action, processed_next_state if not done else None, reward)

            state = processed_next_state
            total_reward += reward
            optimize_model(memory, batch_size, target_net, policy_net, gamma, optimizer)

        epsilon = max(epsilon_limit, epsilon * decay_factor)
        target_net.load_state_dict(policy_net.state_dict())
        avg_reward += total_reward

        print(f"Episode {episode}, Total Reward: {total_reward}")

    env.close()
    torch.save(policy_net.state_dict(), model_path)
    print(f"Model saved to {model_path}")

    avg_reward = avg_reward/episodes
    print(f"AVG_REWARD: {avg_reward}")
    os.makedirs("logs", exist_ok=True)

    # Append results to a CSV log file
    with open("logs/train_log.csv", "a") as log_file:
        log_file.write(f"{model_path},{final_avg:.2f}\n")

def evaluate(model_path = "models/dqn.pth", episodes=1):
    env = gym.make("ALE/Boxing-v5", render_mode="human")
    policy_net = DQN(4, env.action_space.n).to(device)
    if os.path.exists(model_path):
        policy_net.load_state_dict(torch.load(model_path, map_location=device))
        policy_net.eval()
        watch_agent(policy_net, env, episodes)
    else:
        print("No trained model found.")
    env.close()

def watch_agent(policy_net, env, episodes=1):
    for episode in range(episodes):
        state, _ = env.reset()
        frames = deque(maxlen=4)
        frames, state = stack_frames(frames, state, True)
        total_reward = 0
        done = False
        while not done:
            env.render()
            action = select_action(state, policy_net, env.action_space.n, 0.0)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            frames, state = stack_frames(frames, next_state, False)
            total_reward += reward
        print(f"Evaluation Episode {episode}, Total Reward: {total_reward}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--epsilon", type=float, default=1.0, help="Initial epsilon")
    parser.add_argument("--epsilon_limit", type=float, default=0.1, help="Min epsilon")
    parser.add_argument("--decay_factor", type=float, default=0.995, help="Epsilon decay")
    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_path", type=str, default="models/dqn.pth", help="Model path")
    parser.add_argument("--episodes", type=int, default=100, help="Num episodes")

    args = parser.parse_args()

    if args.train:
        train(
            batch_size=args.batch_size,
            gamma=args.gamma,
            epsilon=args.epsilon,
            epsilon_limit=args.epsilon_limit,
            decay_factor=args.decay_factor,
            LR=args.LR,
            model_path=args.model_path,
            episodes=args.episodes
        )
    elif args.eval:
        evaluate(model_path=args.model_path,
                episodes=args.episodes)
    else:
        print("Specify either --train or --eval.")

if __name__ == "__main__":
    main()