import argparse
import torch
from torch import nn
import gym
import numpy as np
from collections import deque
import random
from copy import deepcopy
from gym.wrappers import AtariPreprocessing, FrameStack
from gym import spaces
import matplotlib.pyplot as plt
import csv
## might need to run these commands first

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install box2d pygame


class NeuralNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(input_dim, 16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 9 * 9, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim)
        )

    def forward(self, x: torch.Tensor):
        x = self.network(x)
        return x
    
    
class ReplayBuffer():
    '''Represents replay buffer for storing trajectories, uses
    deque for automatic re-sizing and efficient removing/appending.'''

    def __init__(self, size):
        self.buffer = deque(maxlen=size)

    def add(self, obs: np.ndarray, action: int, reward: float, next_obs: np.ndarray, done: bool):
        obs = np.array(obs, dtype=np.uint8)
        next_obs = np.array(next_obs, dtype=np.uint8)
        action = np.uint8(action)
        reward = np.float16(reward)
        self.buffer.append([obs, action, reward, next_obs, done])

    def sample(self, sample_size):
        if len(self.buffer) < sample_size:
            raise ValueError()    # if value error means sample size too small
        sampled_replays = random.sample(self.buffer, sample_size)
        return sampled_replays


class DQN():
    def __init__(self, env: gym.Env, qnetwork: NeuralNetwork, buffer: ReplayBuffer, hyperparams: dict, device, rewards_path: str):
        self.device = device
        self.env = env
        self.online_model = qnetwork.to(self.device)  ## network for learning q values and selecting actions
        self.target_model = deepcopy(qnetwork).to(self.device)    ## target network for better convergence (only updated periodically)
        self.lr = hyperparams['lr']   ## learning rate
        self.gamma = hyperparams['gamma']   ## discount factor
        self.initial_eps = hyperparams['initial_eps']
        self.eps_decay = hyperparams['eps_decay']
        self.final_eps = hyperparams['final_eps']   ## lower boundary of eps
        self.sample_size = hyperparams['sample_size']
        self.eps = self.initial_eps
        self.buffer = buffer
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(self.online_model.parameters(), lr=self.lr)
        self.update_freq = hyperparams['update_freq']
        self.all_rewards = []
        self.rewards_path = rewards_path


    def eps_greedy(self, obs: torch.Tensor):
        '''Selects an action according to the last 2 observations using epsilon-greedy.'''


        rand = np.random.rand()
        if rand < self.eps:     ## choose random action
            return self.env.action_space.sample()
        else:    ## choose best action
            with torch.no_grad():
                q_values = self.online_model(obs.unsqueeze(0))
                action = torch.argmax(q_values).item()
                # print(q_values)
                # print(action)
                return action

    def train(self, num_frames: int):
        reward_buffer = deque(maxlen=10)  ## buffer for keeping track of rewards over last 10 episodes
        frames = 0
        episodes = 0
        with open(self.rewards_path, mode="w", newline="") as file: # creat a csv file for storage the reward
            writer = csv.writer(file)
            writer.writerow(["episode", "total_reward"])
        
            while frames < num_frames:
                episodes += 1
                obs, info = self.env.reset()   ## get first observation

                terminated = False
                truncated = False
                total_reward = 0
                while not terminated and not truncated:
                    action = self.eps_greedy(torch.tensor(np.array(obs), dtype=torch.float32, device=self.device) / 255.0)
                    # print(action)
                    new_obs, reward, terminated, truncated, info = self.env.step(action) 
                    frames += 1
                    self.buffer.add(np.array(obs), 
                                    action, 
                                    reward, 
                                    np.array(new_obs), 
                                    terminated or truncated)
                    self.update_step()

                    total_reward += reward
                    self.eps = max(self.eps * self.eps_decay, self.final_eps)
                    obs = new_obs

                self.all_rewards.append(total_reward)

                reward_buffer.append(total_reward)
                writer.writerow([episodes, total_reward]) # save the reward

                if episodes % self.update_freq == 0:
                    self.target_model.load_state_dict(self.online_model.state_dict())
                    # print("Target model updated")

                if episodes % 10 == 0:
                    print(f"Episode {episodes} -- Reward Over Last 10 episodes: {np.mean(reward_buffer)}")
                    print(f"Epsilon: {self.eps}")


    def update_step(self):
        '''Uses data from replay buffer to update the model.'''

        if len(self.buffer.buffer) < 1_000:
            return

        try:   
            trajectories = self.buffer.sample(self.sample_size)   ## sample batch from buffer
        except ValueError:   ## if not enough samples in buffer
            return
        
        states = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=self.device) / 255.0 for t in trajectories])
        actions = torch.tensor([t[1] for t in trajectories], device=self.device, dtype=torch.int64)
        rewards = torch.tensor([t[2] for t in trajectories], dtype=torch.float32, device=self.device)
        next_states = torch.stack([torch.tensor(t[3], dtype=torch.float32, device=self.device) / 255.0 for t in trajectories])
        dones = torch.tensor([t[4] for t in trajectories], dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            next_q_values = self.target_model(next_states)
            max_next_q_values = next_q_values.max(dim=1)[0]
            targets = rewards + (1 - dones.float()) * self.gamma * max_next_q_values

        pred = self.online_model(states)
        target = pred.clone().detach()    ## change predictions only at index of action taken, otherwise 
                                                                  ## stay the same as predictions

        target[range(self.sample_size), actions] = targets
        
        loss = self.loss_fn(pred, target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
class ReducedActionWrapper(gym.ActionWrapper):
    def __init__(self, env, allowed_actions):
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = spaces.Discrete(len(allowed_actions))

    def action(self, action):
        return self.allowed_actions[action]
                

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--decay_percentage", type=float, default=0.1, help="Epsilon decay")
    parser.add_argument("--LR", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--model_path", type=str, default="models/dqn.pth", help="Model path")
    parser.add_argument("--rewards_path", type=str, default="rewards/dqn.csv", help="rewards record path")

    args = parser.parse_args()


    
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    final_eps = 0.1
    num_frames = 1_000_000
    
    env = gym.make("ALE/Boxing-v5", obs_type="grayscale", frameskip=1)
    env = AtariPreprocessing(env)   ## Adds automatic frame skipping and frame preprocessing, as well as 
                                    ## starting the environment stochastically by choosing to do nothing
                                    ## for a random number of frames at start
    env = FrameStack(env, num_stack=4)      ## Adds automatic frame stacking for better observability
    # env = ReducedActionWrapper(env, [0, 1, 2, 3, 4, 5])
    network = NeuralNetwork(4, env.action_space.n).to(device)
    # buffer = ReplayBuffer(int(0.1 * num_frames))
    buffer = ReplayBuffer(50_000)

    dqn = DQN(env, network, buffer, {
        'lr': args.LR,
        'gamma': args.gamma,
        'initial_eps': 1.0,
        'eps_decay': np.exp(np.log(final_eps) / (num_frames * args.decay_percentage)),     ## to decay to final_eps after about 10% of training 
        'final_eps': 0.1,
        'sample_size': 32,
        'update_freq': 1 ## how often to update the target network (in terms of episodes)
    }, device, args.rewards_path)
  
    try:
        dqn.train(num_frames)
    except KeyboardInterrupt:   ## allows for automatic saving when training is interrupted
        print("Training interrupted. Saving model...")
        torch.save(dqn.online_model, args.model_path)

    torch.save(dqn.online_model, args.model_path)