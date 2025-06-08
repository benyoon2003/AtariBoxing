import torch
from torch import nn
import gym
import numpy as np
from collections import deque
import random
from copy import deepcopy
import cv2

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

    def add(self, obs: torch.Tensor, action: int, reward: float, new_obs: torch.Tensor, done: bool):
        self.buffer.append([obs, action, reward, new_obs, done])

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
        self.loss_fn = nn.MSELoss()
        self.optimizer = torch.optim.RMSprop(self.online_model.parameters(), lr=self.lr)
        self.update_freq = hyperparams['update_freq']

    def eps_greedy(self, stacked_obs: deque):
        '''Selects an action according to the last 4 observations using epsilon-greedy.'''
        obs = deepcopy(stacked_obs)
        obs = torch.cat(list(stacked_obs)).unsqueeze(0)

        rand = np.random.rand()
        if rand < self.eps:     ## choose random action
            return self.env.action_space.sample()
        else:    ## choose best action
            with torch.no_grad():
                q_values = self.online_model(obs)
                max_indices = torch.where(q_values == q_values.max())[0].cpu().numpy()
                return np.random.choice(max_indices)

    def train(self, num_episodes: int):
        reward_buffer = deque(maxlen=50)

        for episode in range(num_episodes):
            stacked_obs = deque(maxlen=4)    ## keeps last 4 frames

            obs, info = self.env.reset()   ## get first observation
            stacked_obs.append(self._preprocess(obs))

            terminated = False
            truncated = False
            total_reward = 0
            while not terminated and not truncated:
                action = self.eps_greedy(stacked_obs) if len(stacked_obs) == 4 else 0   ## do nothing for first 4 steps
                new_obs, reward, terminated, truncated, info = self.env.step(action)  
                new_stacked_obs = deepcopy(stacked_obs)
                new_stacked_obs.append(self._preprocess(new_obs))
                if len(stacked_obs) == 4:
                        stacked_frames = self._stack_frames(stacked_obs)
                        new_stacked_frames = self._stack_frames(new_stacked_obs)
                        self.buffer.add(stacked_frames, action, reward, new_stacked_frames, terminated or truncated)
                        self.update_step()
                stacked_obs.append(self._preprocess(new_obs))


                total_reward += reward
                self.eps = max(self.eps * self.eps_decay, self.final_eps)
            
            reward_buffer.append(total_reward)
            if episode % self.update_freq == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())

            if episode % (num_episodes // 20) == 0:
                print(f"Episode {episode} -- Average Reward Over Last 50 episodes: {np.mean(reward_buffer)}")

    def _preprocess(self, obs: np.ndarray):
        '''Converts a grayscale observation to a 84 by 84 tensor with values between 0 and 1.'''
        resized = cv2.resize(obs, (84, 84), interpolation=cv2.INTER_AREA)
        resized = resized / 255.0
        return torch.tensor(resized, dtype=torch.float32, device=self.device).unsqueeze(0)
    
    def _stack_frames(self, frames: deque):
        return torch.cat(list(frames))

    def update_step(self):
        '''Uses data from replay buffer to update the model.'''

        try:   
            trajectories = self.buffer.sample(self.sample_size)   ## sample batch from buffer
        except ValueError:   ## if not enough samples in buffer
            return
        
        states = torch.stack([t[0] for t in trajectories])
        actions = torch.tensor([t[1] for t in trajectories], device=self.device, dtype=torch.int32)
        rewards = torch.tensor([t[2] for t in trajectories], dtype=torch.float32, device=self.device)
        next_states = torch.stack([t[3] for t in trajectories])
        dones = torch.tensor([t[4] for t in trajectories], dtype=torch.bool, device=self.device)
        
        with torch.no_grad():
            next_q_values_online = self.online_model(next_states)  ## get q values for next states from online model
            best_actions = torch.argmax(next_q_values_online, dim=1)  ## get best actions for next states
            next_q_values_target = self.target_model(next_states)
            max_q_values_target = next_q_values_target[range(self.sample_size), best_actions]  ## get q values for best actions from target model
            targets = rewards + (1 - dones.float()) * self.gamma * max_q_values_target

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

    num_episodes = 500
    final_eps = 0.1
    average_steps_per_episode = 1_000
    
    env = gym.make("ALE/Pong-v5", obs_type="grayscale")
    network = NeuralNetwork(env.action_space.n).to(device)
    buffer = ReplayBuffer(int(0.1 * num_episodes * average_steps_per_episode))
    # buffer = ReplayBuffer(1_000)

    dqn = DQN(env, network, buffer, {
        'lr': 0.00005,
        'gamma': 0.995,
        'initial_eps': 1,
        'eps_decay': np.exp(np.log(final_eps) / (num_episodes * .75 * average_steps_per_episode)),     ## to decay to final_eps after about 75% of training
        # 'eps_decay': 0.995,  
        'final_eps': final_eps,
        'sample_size': 32,
        'update_freq': 1   ## how often to update the target network (in terms of episodes)
    }, device)

    dqn.train(num_episodes)


    torch.save(dqn.online_model, "./DQN/dqn_model.pth")