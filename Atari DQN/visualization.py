import gym
import numpy as np
import torch
from ram_train import NeuralNetwork

env = gym.make("ALE/Breakout-v5", obs_type="ram", render_mode='human')

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = torch.load("./Atari DQN/breakout_dqn.pth").to(device)
# model = torch.load("dqn_model_v2.pth").to(device)
model.eval()

obs, info = env.reset()
all_rewards = []
all_steps = []
for _ in range(100):
    obs, info = env.reset()
    terminated = False
    truncated = False
    rewards = 0
    steps = 0
    obs, reward, terminated, truncated, info = env.step(1)
    while not terminated and not truncated:
        with torch.no_grad():
            q_values = model(torch.tensor(obs, dtype=torch.float32, device=device))
            max_indices = torch.where(q_values == q_values.max())[0].cpu().numpy()
            action = np.random.choice(max_indices)
        obs, reward, terminated, truncated, info = env.step(action)
        # print(action)
        rewards += reward
        steps += 1

    print(rewards)
    all_rewards.append(rewards)
    all_steps.append(steps)

print(np.mean(all_rewards))
print(np.mean(all_steps))


# num_episodes = 300
# final_eps = 0.1
# average_steps_per_episode = 150

# print(np.exp(np.log(final_eps) / (num_episodes * .75 * average_steps_per_episode)))