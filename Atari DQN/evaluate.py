import gym
import numpy as np
import torch
from pixels_train_v2 import NeuralNetwork
from gym.wrappers import AtariPreprocessing, FrameStack
    

env = gym.make("ALE/Boxing-v5", obs_type="grayscale", frameskip=1)
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = torch.load("./Atari DQN/Models/double_dqn.pth").to(device)
model.eval()

obs, info = env.reset()
all_rewards = []
for _ in range(20):
    obs, info = env.reset()
    terminated = False
    truncated = False
    rewards = 0
    while not terminated and not truncated:
        with torch.no_grad():
            q_values = model(torch.tensor(np.array(obs), dtype=torch.float32, device=device).unsqueeze(0) / 255.0)
            action = torch.argmax(q_values).item()
        obs, reward, terminated, truncated, info = env.step(action)
        rewards += reward

    all_rewards.append(rewards)

print(np.mean(all_rewards))