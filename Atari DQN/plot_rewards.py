import pandas as pd
import matplotlib.pyplot as plt

# Load the CSV data
df = pd.read_csv("Atari DQN/rewards3.csv")  # Save your data with this filename

# Plot total reward per episode
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"], linestyle='-')
plt.title("Episode Rewards Over Time")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()