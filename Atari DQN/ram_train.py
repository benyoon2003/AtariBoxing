from copy import deepcopy
import torch
from torch import nn
import gym
import numpy as np
from collections import deque
import random
from gym.wrappers import AtariPreprocessing

## might need to run these commands first

# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
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

    def forward(self, x: torch.Tensor):
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
        self.online_model = qnetwork.to(self.device)  ## network for learning q values
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

    def eps_greedy(self, obs: torch.Tensor):
        '''Selects an action using epsilon-greedy.'''

        obs = obs.to(self.device)
        rand = np.random.rand()
        if rand < self.eps:     ## choose random action
            return self.env.action_space.sample()
        else:    ## choose best action
            with torch.no_grad():
                q_values = self.online_model(obs.unsqueeze(0))
                q_values = q_values.cpu().numpy().squeeze()  
                # print(q_values)
                max_indices = np.where(q_values == q_values.max())[0]
                action = np.random.choice(max_indices)  ## choose one of the best actions randomly
                # print(action)
                return action

    def train(self, num_frames):
        reward_buffer = deque(maxlen=10)
        frames = 0
        episodes = 0
        while frames < num_frames:
            episodes += 1
            obs, info = self.env.reset()   ## get first observation
            terminated = False
            truncated = False
            total_reward = 0
            while not terminated and not truncated:
                action = self.eps_greedy(torch.tensor(obs, dtype=torch.float32, device=self.device))
                # print(action)
                new_obs, reward, terminated, truncated, info = self.env.step(action)  
                frames += 1
                total_reward += reward
                self.buffer.add(obs, action, reward, new_obs, terminated or truncated)
                self.update_step()
                self.eps = max(self.eps * self.eps_decay, self.final_eps)
                obs = new_obs
            
            reward_buffer.append(total_reward)

            if episodes % self.update_freq == 0:
                self.target_model.load_state_dict(self.online_model.state_dict())

            if episodes % 10 == 0:
                print(f"Episode {episodes} -- Average Reward Over Last 10 episodes: {np.mean(reward_buffer)}")
                print(f"Current Epsilon: {self.eps}")


    def update_step(self):
        '''Uses data from replay buffer to update the model.'''

        try:   
            trajectories = self.buffer.sample(self.sample_size)   ## sample batch from buffer
        except TypeError:   ## if not enough samples in buffer
            return
        
        states = torch.stack([torch.tensor(t[0], dtype=torch.float32, device=self.device) for t in trajectories])
        actions = torch.tensor([t[1] for t in trajectories], device=self.device, dtype=torch.int32)
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

    # num_episodes = 1_000
    final_eps = 0.1
    num_frames = 750_000
    # average_steps_per_episode = 150
    
    env = gym.make("ALE/Boxing-v5", obs_type="ram")  ## using RAM observations instead of pixels
    network = NeuralNetwork(128, env.action_space.n).to(device)
    buffer = ReplayBuffer(int(0.1 * num_frames))
    # buffer = ReplayBuffer(10_000)

    dqn = DQN(env, network, buffer, {
        'lr': 0.0005,
        'gamma': 0.99,
        'initial_eps': 1,
        'eps_decay': np.exp(np.log(final_eps) / (num_frames * .75)),     ## to decay to final_eps after about 75% of training
        'final_eps': final_eps,
        'sample_size': 32,
        'update_freq': 5
    }, device)

    try:
        dqn.train(num_frames)
    except KeyboardInterrupt:
        torch.save(dqn.online_model, "./Atari DQN/boxing_dqn.pth")

    torch.save(dqn.online_model, "./Atari DQN/boxing_dqn.pth")