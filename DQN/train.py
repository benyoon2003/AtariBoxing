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
    '''Represents replay buffer for storing trajectories, uses
    deque for automatic re-sizing and efficient removing/appending.'''

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, obs: np.ndarray, action: int, reward: float, new_obs: np.ndarray, terminated: bool):
        self.buffer.append([obs, action, reward, new_obs, terminated])

    def sample(self, sample_size):
        if len(self.buffer) < sample_size:
            raise TypeError()    # if type error means sample size too small
        sampled_replays = random.sample(self.buffer, sample_size)
        return sampled_replays


class DQN():
    def __init__(self, env: gym.Env, qnetwork: NeuralNetwork, buffer: ReplayBuffer, hyperparams: dict, device):
        self.device = device
        self.env = env
        self.model = qnetwork.to(self.device)  ## network for learning q values
        self.lr = hyperparams['lr']   ## learning rate
        self.gamma = hyperparams['gamma']   ## discount factor
        self.initial_eps = hyperparams['initial_eps']
        self.eps_decay = hyperparams['eps_decay']
        self.final_eps = hyperparams['final_eps']   ## lower boundary of eps
        self.sample_size = hyperparams['sample_size']
        self.eps = self.initial_eps
        self.buffer = buffer
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=self.lr)

    def eps_greedy(self, obs: torch.Tensor):
        '''Selects an action using epsilon-greedy.'''

        obs = obs.to(self.device)
        rand = np.random.rand()
        if rand < self.eps:     ## choose random action
            return self.env.action_space.sample()
        else:    ## choose best action
            with torch.no_grad():
                q_values = self.model(obs)
                max_indices = torch.where(q_values == q_values.max())[0].cpu().numpy()
                return np.random.choice(max_indices)

    def train(self, num_episodes):
        for episode in range(num_episodes):
            obs, info = self.env.reset()   ## get first observation
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action = self.eps_greedy(torch.tensor(obs, dtype=torch.float32))
                new_obs, reward, terminated, truncated, info = self.env.step(action)  
                self.buffer.add(obs, action, reward, new_obs, terminated)
                self.update_step()
                self.eps = max(self.eps * self.eps_decay, self.final_eps)
                obs = new_obs

            if episode % (num_episodes // 100) == 0:
                print(f"Episode {episode} -- Averarge Reward: {self.evaluate()}")


    def update_step(self):
        '''Uses data from replay buffer to update the model.'''

        try:   
            trajectories = self.buffer.sample(self.sample_size)   ## sample batch from buffer
        except TypeError:   ## if not enough samples in buffer
            return
        
        states = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in trajectories]).to(self.device)
        actions = torch.tensor([t[1] for t in trajectories]).to(self.device)
        rewards = torch.tensor([t[2] for t in trajectories], dtype=torch.float32).to(self.device)
        next_states = torch.stack([torch.tensor(t[3], dtype=torch.float32) for t in trajectories]).to(self.device)
        dones = torch.tensor([t[4] for t in trajectories], dtype=torch.bool).to(self.device)
        
        with torch.no_grad():
            next_q_values = self.model(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards + (1 - dones.float()) * self.gamma * max_next_q_values

        pred = self.model(states)
        target = pred.clone().detach()    ## change predictions only at index of action taken, otherwise 
                                                                  ## stay the same as predictions

        target[range(self.sample_size), actions] = targets


        loss = self.loss_fn(pred, target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
            

        # for trajectory in trajectories:
        #     state = torch.tensor(trajectory[0], dtype=torch.float32).to(self.device)
        #     next_state = torch.tensor(trajectory[3], dtype=torch.float32).to(self.device)
        #     pred = self.model(state)  ## get predicted q-values
        #     with torch.no_grad():
        #         next_pred = self.model(next_state)  ## get next predicted q-values

        #         max_indices = torch.where(next_pred == next_pred.max())[0].cpu().numpy()
        #         a_prime = np.random.choice(max_indices)
            
        #         target = pred.clone().detach()
        #         target = target.to(self.device)

        #     if trajectory[4]:    ## if the episode is done
        #         target[trajectory[1]] = trajectory[2]   ## target is just the reward for the given action
        #     else:    ## the episode is not done
        #         target[trajectory[1]] = trajectory[2] + self.gamma * next_pred[a_prime]     ## bellman equation 

        #     loss = self.loss_fn(pred, target)
        #     loss.backward()
        #     self.optimizer.step()
        #     self.optimizer.zero_grad()

    def evaluate(self):
        all_rewards = []
        for episode in range(100):
            obs, info = self.env.reset()
            terminated = False
            truncated = False
            total_reward = 0
            while not terminated and not truncated:
                with torch.no_grad():
                    obs_tensor = torch.tensor(obs, dtype=torch.float32).to(self.device)
                    q_values = self.model(obs_tensor)
                    max_indices = torch.where(q_values == q_values.max())[0].cpu().numpy()
                    action = np.random.choice(max_indices)
                new_obs, reward, terminated, truncated, info = self.env.step(action)
                total_reward += reward
                obs = new_obs

            all_rewards.append(total_reward)
        return np.mean(all_rewards)
                

if __name__ == "__main__":
    device = device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    
    env = gym.make("LunarLander-v2")
    network = NeuralNetwork(8, env.action_space.n).to(device)
    buffer = ReplayBuffer(10000)

    num_episodes = 10_000
    final_eps = 0.1
    dqn = DQN(env, network, buffer, {
        'lr': 0.001,
        'gamma': 0.99,
        'initial_eps': 1,
        'eps_decay': np.exp(np.log(final_eps) / (num_episodes * 500 / 2)),     ## to decay to final_eps after about 50% of training
        'final_eps': final_eps,
        'sample_size': 32
    }, device)

    dqn.train(num_episodes)
    torch.save(dqn.model, "dqn_model")