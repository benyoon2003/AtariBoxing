import gym
import numpy as np
import torch
# from ram_train import NeuralNetwork
from pixels_train import NeuralNetwork
from gym.wrappers import AtariPreprocessing, FrameStack
from gym import spaces


class ReducedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = spaces.Discrete(len(allowed_actions))

    def action(self, action):
        return self.allowed_actions[action]
    

env = gym.make("ALE/Boxing-v5", obs_type="grayscale", frameskip=1)
env = AtariPreprocessing(env)
env = FrameStack(env, num_stack=4)
# env = ReducedActionWrapper(env, [0, 1, 2, 3, 4, 5])

# env = gym.make("ALE/Boxing-v5", obs_type="ram", render_mode='human')


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

model = torch.load("dqn3.pth").to(device)
# model = torch.load("dqn_model_v2.pth").to(device)
model.eval()

obs, info = env.reset()
all_rewards = []
all_steps = []
for _ in range(3):
    obs, info = env.reset()
    terminated = False
    truncated = False
    rewards = 0
    steps = 0
    obs, reward, terminated, truncated, info = env.step(1)
    while not terminated and not truncated:
        with torch.no_grad():
            q_values = model(torch.tensor(np.array(obs), dtype=torch.float32, device=device).unsqueeze(0) / 255.0)
            q_values = q_values.cpu().numpy().squeeze()
            # print(q_values)
            max_indices = np.where(q_values == q_values.max())[0]
            # print(max_indices)
            action = np.random.choice(max_indices)
            # action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        # print(action)
        rewards += reward
        steps += 1

    # print(rewards)
    all_rewards.append(rewards)
    all_steps.append(steps)

print(np.mean(all_rewards))
print(np.mean(all_steps))


# num_episodes = 300
# final_eps = 0.1
# average_steps_per_episode = 150

# print(np.exp(np.log(final_eps) / (num_episodes * .75 * average_steps_per_episode)))