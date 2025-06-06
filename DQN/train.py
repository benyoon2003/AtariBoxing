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
    def __init__(self, size): # size = 1000
        self.buffer = deque(maxlen=size) # use deque to easy remove from top and append from bottom

    def add(self, obs: np.ndarray, action: int, reward: int, new_obs: np.ndarray, terminated: bool):
        self.buffer.append([obs, action, reward, new_obs, terminated])

    def sample(self, sample_size):
        if len(self.buffer) < sample_size:
            raise TypeError() # if type error mean you sample size too small
        sampled_replays = random.sample(list(self.buffer), sample_size) # select random number(sample_size) from the buffer
        return sampled_replays


class DQN():
    def __init__(self, env: gym.Env, qnetwork: NeuralNetwork, buffer: ReplayBuffer, hyperparams: dict):
        self.env = env
        self.model = qnetwork # the neuralNetwork you wana use
        self.lr = hyperparams['lr'] # learning rate
        self.gamma = hyperparams['gamma'] # discount factor
        self.initial_eps = hyperparams['initial_eps'] # initial eps = 1
        self.eps_decay = hyperparams['eps_decay'] 
        self.final_eps = hyperparams['final_eps'] # lower boundary of eps
        self.sample_size = hyperparams['sample_size'] # sample_size
        self.eps = self.initial_eps
        self.buffer = buffer 

    def eps_greedy(self, obs): # select action
        rand = np.random.rand() # choise a random number from 0 to 1
        if rand < self.eps: 
            return self.env.action_space.sample() #random select a action
        else:
            q_values = self.model(obs) # put current obs to model to get q value
            return np.argmax(q_values) # return max score of q value

    def train(self, num_episodes):
        for episode in range(num_episodes):
            obs, info = self.env.reset() # find current location of s
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = self.eps_greedy(obs)
                new_obs, reward, terminated, truncated, info = self.env.step(action) # get new reward and new state
                self.buffer.add(obs, action, reward, new_obs, terminated) # append it to the replay buffer
                obs = new_obs # update current state
                print(obs, reward, action) 
                self.update_step()
                self.eps = max(self.eps * self.eps_decay, 0.05)
                obs = new_obs

            # if episode % (num_episodes / 20) == 0:
                # self.evaluate()


    ### need to fix this method
    def update_step(self): # use the data from replaybuffer to upgrade the model

        loss_fn = nn.MSELoss()
        optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)
        try:
            trajectories = self.buffer.sample(self.sample_size) # take random function form the replay buffer
        except TypeError:
            return
        for trajectory in trajectories:
            trajectory = np.array(trajectory)
            
            # use bellman fucntion to calculate the target
            pred = self.model(torch.Tensor(trajectory[0]))
            next_pred = self.model(torch.Tensor(trajectory[3]))
            next_pred = next_pred.numpy()
            max_indices = np.flatnonzero(next_pred == np.max(next_pred))
            a_prime = np.random.choice(max_indices)
            target = np.copy(pred)
            if trajectory[4]:
                target[trajectory[1]] = trajectory[2]
            else:
                target[trajectory[1]] = trajectory[2] + self.gamma * next_pred[a_prime]
            # get loss value from pred and target
            loss = loss_fn(pred, target)
            loss.backward()
            optimizer.step()# upgrade optizer weight
            optimizer.zero_grad()# uprade optimizer grad

    # def evaluate(self):
    #     self.model.eval()
    #     for episode in range(100):
    #         obs, info = self.env.reset()
    #         terminated = False
    #         truncated = False
    #         while not terminated and not truncated:
                

if __name__ == "__main__":
    env = gym.make("LunarLander-v2")
    network = NeuralNetwork(8, env.action_space.n)
    buffer = ReplayBuffer(1000)
    dqn = DQN(env, network, buffer, {
        'lr': 0.01,
        'gamma': 0.99,
        'initial_eps': 1,
        'eps_decay': 0.999,
        'final_eps': 0.1,
        'sample_size': 32
    })

    dqn.train(100)