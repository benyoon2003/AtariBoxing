import pandas as pd
import matplotlib.pyplot as plt
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Plot DQN rewards from a CSV file.")
parser.add_argument("csv_path", type=str, help="Path to the CSV file containing rewards.")
args = parser.parse_args()

# Load the CSV data
df = pd.read_csv(args.csv_path)

# Plot total reward per episode
plt.figure(figsize=(10, 5))
plt.plot(df["episode"], df["total_reward"], linestyle='-')
plt.title(f"Episode Rewards Over Time for {args.csv_path[8:-4]}")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.show()

second_column = df.iloc[:, 1]
average_value = second_column.mean()

print("Average Rewards:", average_value)