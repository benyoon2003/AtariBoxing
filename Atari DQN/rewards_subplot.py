import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

paths = [
    "Atari DQN/Rewards/dqn_lr0.0001_gamma0.98_decay0.1_850000frames.csv",
    "Atari DQN/Rewards/dqn_lr0.00001_gamma0.98_decay0.1_850000frames.csv",
    "Atari DQN/Rewards/dqn_lr0.0001_gamma0.98_decay0.2_850000frames.csv",
    "Atari DQN/Rewards/dqn_lr0.00001_gamma0.98_decay0.2_850000frames.csv",
    "Atari DQN/Rewards/dqn_lr0.00001_gamma0.995_decay0.2_850000frames.csv"
]

plt.figure(figsize=(10, 7))
df = pd.read_csv(paths[0])
df['smoothed_reward'] = df['total_reward'].ewm(span=80).mean()
plt.plot(df['episode'], df['smoothed_reward'], label="LR 0.0001 | Gamma 0.98 | Eps Decay Duration 0.1")

df = pd.read_csv(paths[1])
df['smoothed_reward'] = df['total_reward'].ewm(span=80).mean()
plt.plot(df['episode'], df['smoothed_reward'], label="LR 0.00001 | Gamma 0.98 | Eps Decay Duration 0.1")

df = pd.read_csv(paths[2])
df['smoothed_reward'] = df['total_reward'].ewm(span=80).mean()
plt.plot(df['episode'], df['smoothed_reward'], label="LR 0.0001 | Gamma 0.98 | Eps Decay Duration 0.2")

df = pd.read_csv(paths[3])
df['smoothed_reward'] = df['total_reward'].ewm(span=80).mean()
plt.plot(df['episode'], df['smoothed_reward'], label="LR 0.00001 | Gamma 0.98 | Eps Decay Duration 0.2")

df = pd.read_csv(paths[4])
df['smoothed_reward'] = df['total_reward'].ewm(span=80).mean()
plt.plot(df['episode'], df['smoothed_reward'], label="LR 0.00001 | Gamma 0.995 | Eps Decay Duration 0.2")

plt.title(f"Episode Rewards During Training")
plt.xlabel("Episode")
plt.ylabel("Smoothed Episode Rewards")
plt.legend(loc='upper left')
plt.show()
    