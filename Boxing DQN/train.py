import torch
from torch import nn
import gym
import numpy as np
from collections import deque
import random
from copy import deepcopy
from gym.wrappers import AtariPreprocessing, FrameStack

## might need to run these commands first

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
# pip install box2d pygame


class NeuralNetwork(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=8, stride=4),
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

    def add(self, obs: np.array, action: int, reward: float, next_obs: np.array, done: bool):
        self.buffer.append([obs, action, reward, next_obs, done])

    def sample(self, sample_size):
        if len(self.buffer) < sample_size:
            raise ValueError()    # if value error means sample size too small
        sampled_replays = random.sample(self.buffer, sample_size)
        return sampled_replays


class DQN():
    def __init__(self, env: gym.Env, qnetwork: NeuralNetwork, buffer: ReplayBuffer, hyperparams: dict, device):
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
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(), lr=self.lr)
        self.update_freq = hyperparams['update_freq']

    def eps_greedy(self, obs: torch.Tensor):
        '''Selects an action according to the last 4 observations using epsilon-greedy.'''

        rand = np.random.rand()
        if rand < self.eps:     ## choose random action
            return self.env.action_space.sample()
        else:    ## choose best action
            with torch.no_grad():
                q_values = self.online_model(obs.unsqueeze(0))
                max_indices = torch.where(q_values == q_values.max())[0].cpu().numpy()
                return np.random.choice(max_indices)

    def train(self, num_episodes: int):
        reward_buffer = deque(maxlen=10)  ## buffer for keeping track of rewards over last 10 episodes

        for episode in range(num_episodes):

            obs, info = self.env.reset()   ## get first observation

            terminated = False
            truncated = False
            total_reward = 0
            while not terminated and not truncated:
                action = self.eps_greedy(torch.tensor(np.array(obs), dtype=torch.float32, device=self.device) / 255.0)
                new_obs, reward, terminated, truncated, info = self.env.step(action) 
                self.buffer.add(np.array(obs) / 255.0, 
                                action, 
                                reward, 
                                np.array(new_obs) / 255.0, 
                                terminated or truncated)
                self.update_step()

                total_reward += reward
            self.eps = max(self.eps * self.eps_decay, self.final_eps)
            
            reward_buffer.append(total_reward)

            if episode % self.update_freq == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())

            if episode % (num_episodes // 20) == 0:
                print(f"Episode {episode} -- Reward Over Last 10 episodes: {np.mean(reward_buffer)}")
                print(f"Epsilon: {self.eps}")


    def update_step(self):
        '''Uses data from replay buffer to update the model.'''

        try:   
            trajectories = self.buffer.sample(self.sample_size)   ## sample batch from buffer
        except ValueError:   ## if not enough samples in buffer
            return
        
        states = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=self.device) for t in trajectories])
        actions = torch.tensor([t[1] for t in trajectories], device=self.device, dtype=torch.int64)
        rewards = torch.tensor([t[2] for t in trajectories], dtype=torch.float32, device=self.device)
        next_states = torch.stack([torch.tensor(t[3], dtype=torch.float32, device=self.device) for t in trajectories])
        dones = torch.tensor([t[4] for t in trajectories], dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            next_q_values_online = self.online_model(next_states)  ## get q values for next states from online model
            best_actions = torch.argmax(next_q_values_online, dim=1)  ## get best actions for next states
            next_q_values_target = self.target_model(next_states)
            max_q_values_target = next_q_values_target[range(self.sample_size), best_actions]  ## get q values for best actions from target model
            targets = rewards + (1 - dones.float()) * self.gamma * max_q_values_target

        # print(states.shape)
        # print(states)
        pred = self.online_model(states)
        target = pred.clone().detach()    ## change predictions only at index of action taken, otherwise 
                                                                  ## stay the same as predictions

        target[range(self.sample_size), actions] = targets
        
        loss = self.loss_fn(pred, target)


        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        torch.cuda.empty_cache()  ## clear cache to avoid memory issues, especially on GPU
    
                

if __name__ == "__main__":
    device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
    )
    print(f"Using {device} device")

    # print("Torch version:", torch.__version__)
    # print("CUDA available:", torch.cuda.is_available())
    # print("GPU name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU")

    num_episodes = 1000
    final_eps = 0.1
    average_steps_per_episode = 1_000
    
    env = gym.make("ALE/Pong-v5", obs_type="grayscale", frameskip=1)
    env = AtariPreprocessing(env)   ## Adds automatic frame skipping and frame preprocessing, as well as 
                                    ## starting the environment stochastically by choosing to do nothing
                                    ## for a random number of frames at start
    env = FrameStack(env, num_stack=4)      ## Adds automatic frame stacking for better observability
    network = NeuralNetwork(env.action_space.n).to(device)
    buffer = ReplayBuffer(int(0.1 * num_episodes * average_steps_per_episode))
    # buffer = ReplayBuffer(3_000)

    dqn = DQN(env, network, buffer, {
        'lr': 0.0001,
        'gamma': 0.99,
        'initial_eps': 1.0,
        # 'eps_decay': np.exp(np.log(final_eps) / (num_episodes * .5 * average_steps_per_episode)),     ## to decay to final_eps after about 50% of training
        # 'eps_decay': 0.995,  
        'eps_decay': np.exp(np.log(final_eps) / (num_episodes * 0.5)),     ## to decay to final_eps after about 50% of training
        'final_eps': final_eps,
        'sample_size': 32,
        'update_freq': 1  ## how often to update the target network (in terms of episodes)
    }, device)

    dqn.train(num_episodes)


    torch.save(dqn.online_model, "./Boxing DQN/dqn_model.pth")