import gymnasium as gym
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import copy
import random
import numpy as np
from collections import deque
import multiprocessing as mp
import os
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

ENV_NAME = "ALE/Boxing-v5"
env = gym.make(ENV_NAME, render_mode=None)

# CNN grabs game state
class CNN(nn.Module):
    def __init__(self, action_size):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.out = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
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


# Fitness of the policy is determined based on average total reward per episode
def evaluate_policy(policy, n_episodes=1, render=False, print_reward=False):
    policy.eval()
    env_local = gym.make(ENV_NAME, render_mode="human" if render else None)
    total_reward = 0.0

    # Iterate through episodes
    for _ in range(n_episodes):
        frames = deque(maxlen=4)
        obs, _ = env_local.reset()
        frames, state = stack_frames(frames, obs, True) # Grab stacked frames for current state
        done = False
        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = policy(state_tensor)
                action = torch.argmax(logits, dim=1).item() # Grab action

            next_obs, reward, terminated, truncated, _ = env_local.step(action)
            done = terminated or truncated
            frames, state = stack_frames(frames, next_obs, False) # Grab next state
            total_reward += reward # Collect reward
            if (render):
                env_local.render()

    avg_reward = total_reward / n_episodes
    if print_reward:
        print(f"Average reward per episode {avg_reward}")
    return avg_reward

# Select policy based on fitness score
def select_elites(population, fitnesses, elite_frac):
    elites_select = max(1, int(len(population) * elite_frac)) # Select the top elite_frac percent. needs to be at least 1
    sorted_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True) # Sort fitnesses in descending order while maintaining index values
    elites = [population[i] for i in sorted_indices[:elites_select]] # extract the actual elite policies
    return elites


# Randomly combine features of two parents to produce child
def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    with torch.no_grad():
        for p1, p2, c in zip(parent1.parameters(), parent2.parameters(), child.parameters()): # iterate through weights and biases of parents
            # Copy each element from p1 or p2 with 50% probability
            for i in range(p1.numel()):
                if random.random() > 0.5:
                    c.view(-1)[i] = p1.view(-1)[i] # flatten so we can access
                else:
                    c.view(-1)[i] = p2.view(-1)[i]
    return child.to(device)

# Mutate some parts of the weights and biases with Gaussian noise
def mutate(network, mutation_rate=0.01, mutation_strength=0.1):
    with torch.no_grad():
        for param in network.parameters():
            if random.random() < mutation_rate:
                noise = torch.randn_like(param) * mutation_strength
                param.add_(noise)

# Wrapper so we can do some parallel computing
def evaluate_worker(args):
    idx, policy, episodes = args
    fitness = evaluate_policy(policy, n_episodes=episodes)
    return idx, fitness

def load_policy(model_path, action_size):
    model = CNN(action_size).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

# Trains using genetic algorithm
def train(pop_size, generations, elite_frac, mutation_rate, mutation_strength, episodes_per_eval, action_size, model_path):
    
    # Initialize population and fitness list
    population = [CNN(action_size).to(device) for _ in range(pop_size)]
    fitnesses = [0] * pop_size
    avg_cumulative_fitness = 0

    iterations = []
    values = []

    for gen in range(generations):
        print(f"Generation {gen+1}/{generations}")

        # Evaluate population fitness in parallel
        with mp.Pool(mp.cpu_count()) as pool:
            results = pool.map(
                evaluate_worker,
                [(i, population[i], episodes_per_eval) for i in range(pop_size)]
            )
        for idx, fit in results:
            fitnesses[idx] = fit

        # Get best fitness score and avg scores
        best_fitness = max(fitnesses)
        avg_fitness = sum(fitnesses) / len(fitnesses)
        avg_cumulative_fitness += avg_fitness
        print(f" Best Fitness: {best_fitness:.2f}, Average Fitness: {avg_fitness:.2f}")

        iterations.append(gen)
        values.append(avg_fitness)

        elites = select_elites(population, fitnesses, elite_frac)

        # Create next generation
        next_population = elites.copy()
        while len(next_population) < pop_size:
            parents = random.sample(elites, 2) # randomly select two parents from elites list
            child = crossover(parents[0], parents[1])
            mutate(child, mutation_rate, mutation_strength)
            next_population.append(child)

        population = next_population

    # Save best network
    best_net = population[np.argmax(fitnesses)]
    os.makedirs("models", exist_ok=True)
    torch.save(best_net.state_dict(), model_path)
    print(f"Best model saved to {model_path}")
    with open("logs/train_log.csv", "a") as log_file:
        log_file.write(f"{model_path},{avg_cumulative_fitness/generations}\n")

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, values, marker='o', linestyle='-', color='b', label='Value')
    plt.title("Fitness over Generations")
    plt.xlabel("Generation")
    plt.ylabel("Fitness (score against base agent)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Full file path to save
    os.makedirs("plots/", exist_ok=True)
    plot_path = os.path.join("plots/", model_path[10:-3] + ".png")
    plt.savefig(plot_path)

    return best_net


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--pop_size", type=int, default=20, help="Population Size")
    parser.add_argument("--generations", type=int, default=10, help="Number of generations")
    parser.add_argument("--elite_frac", type=float, default=0.2, help="Grab this percent of the elites for selection")
    parser.add_argument("--mutation_rate", type=float, default=0.1, help="Mutation rate")
    parser.add_argument("--mutation_strength", type=float, default=0.02, help="Mutation Strength")
    parser.add_argument("--episodes_per_eval", type=int, default=3, help="Episodes per eval")
    parser.add_argument("--model_path", type=str, default="models/GA/ga.pth", help="Model path")
    parser.add_argument("--render", type=bool, default=False, help="Render the view")
 
    args = parser.parse_args()

    if args.train:
        best_model = train(
            pop_size=args.pop_size,
            generations=args.generations,
            elite_frac=args.elite_frac,
            mutation_rate=args.mutation_rate,
            mutation_strength=args.mutation_strength,
            episodes_per_eval=args.episodes_per_eval,
            action_size=env.action_space.n,
            model_path=args.model_path
        )
    elif args.eval:
        loaded_policy = load_policy(args.model_path, env.action_space.n)
        evaluate_policy(loaded_policy, n_episodes=args.episodes_per_eval, render=args.render, print_reward=True)
    else:
        print("Specify either --train or --eval.")

if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    main()