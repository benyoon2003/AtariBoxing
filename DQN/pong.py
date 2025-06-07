import gym
import numpy as np
import torch
import cv2

env = gym.make("ALE/Pong-v5", obs_type="grayscale")

obs, info = env.reset()
all_rewards = []
all_steps = []
for _ in range(100):
    obs, info = env.reset()
    terminated = False
    truncated = False
    rewards = 0
    steps = 0
    while not terminated and not truncated:
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frame_resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        # print(frame_resized / 255)
        rewards += reward
        steps += 1



    all_rewards.append(rewards)
    all_steps.append(steps)

print(np.mean(all_rewards))
print(np.mean(all_steps))


# num_episodes = 300
# final_eps = 0.1
# average_steps_per_episode = 150

# print(np.exp(np.log(final_eps) / (num_episodes * .75 * average_steps_per_episode)))