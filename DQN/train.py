import torch
from torch import nn
import gym
import numpy as np
from collections import deque
import random

# from gymnasium.envs import box2d



# pip install box2d pygame






class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )

    def forward(self, x):
        x = self.linear_relu_stack(x)
        return x
    
class ReplayBuffer():
    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, obs: np.ndarray, action: int, reward: float, new_obs: np.ndarray, terminated: bool):
        self.buffer.append([obs, action, reward, new_obs, terminated])

    def sample(self, sample_size):
        if len(self.buffer) < sample_size:
            raise TypeError()
        sampled_replays = random.sample(self.buffer, sample_size)
        return sampled_replays


class DQN():
    def __init__(self, env: gym.Env, qnetwork: NeuralNetwork, buffer: ReplayBuffer, hyperparams: dict):
        self.env = env
        self.model = qnetwork
        self.lr = hyperparams['lr']
        self.gamma = hyperparams['gamma']
        self.initial_eps = hyperparams['initial_eps']
        self.eps_decay = hyperparams['eps_decay']
        self.final_eps = hyperparams['final_eps']
        self.sample_size = hyperparams['sample_size']
        self.eps = self.initial_eps
        self.buffer = buffer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)

    def eps_greedy(self, obs: torch.Tensor):
        rand = np.random.rand()
        if rand < self.eps:
            return self.env.action_space.sample()
        else:
            with torch.no_grad():
                q_values = self.model(obs)
                max_indices = torch.where(q_values == q_values.max())[0]
                return np.random.choice(max_indices)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            obs, info = self.env.reset()
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = self.eps_greedy(torch.Tensor(obs))
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                self.buffer.add(obs, action, reward, new_obs, terminated)
                self.update_step()
                self.eps = max(self.eps * self.eps_decay, self.final_eps)
                obs = new_obs

            if episode % (num_episodes / 20) == 0:
                print(f"Episode {episode} -- Averarge Reward: {self.evaluate()}")


    ### need to fix this method
    def update_step(self):
        try:   
            trajectories = self.buffer.sample(self.sample_size)
        except TypeError:   ## if not enough samples in buffer
            return
        

        for trajectory in trajectories:
            pred = self.model(torch.Tensor(trajectory[0]))
            with torch.no_grad():
                next_pred = self.model(torch.Tensor(trajectory[3]))

                max_indices = torch.where(next_pred == next_pred.max())[0]
                a_prime = np.random.choice(max_indices)
            
                target = pred.clone().detach()

            if trajectory[4]:    ## if the episode is done
                target[trajectory[1]] = trajectory[2]   ## target is just the reward for the given action
            else:    ## the episode is not done
                target[trajectory[1]] = trajectory[2] + self.gamma * next_pred[a_prime] 

            loss = self.loss_fn(pred, target)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

    def evaluate(self):
        all_rewards = []
        for episode in range(100):
            obs, info = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            while not terminated and not truncated:
                with torch.no_grad():
                    q_values = self.model(torch.Tensor(obs))
                    max_indices = torch.where(q_values == q_values.max())[0]
                    action = np.random.choice(max_indices)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                obs = new_obs

            all_rewards.append(total_reward)
        return np.mean(all_rewards)
                

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    network = NeuralNetwork(8, env.action_space.n)
    buffer = ReplayBuffer(100_000)
    dqn = DQN(env, network, buffer, {
        'lr': 0.001,
        'gamma': 0.99,
        'initial_eps': 1,
        'eps_decay': 0.99999,
        'final_eps': 0.1,
        'sample_size': 32
    })

    dqn.train(1_000_000)
    torch.save(dqn.model, "dqn_model")
