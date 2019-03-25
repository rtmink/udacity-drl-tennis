import numpy as np
import random

import torch
import torch.nn.functional as F
import torch.optim as optim

from ou_noise import OUNoise
from model import Actor, Critic
from replay_buffer import ReplayBuffer

BUFFER_SIZE = int(1e6)   # replay buffer size
BATCH_SIZE = 128         # minibatch size
GAMMA = 0.99             # discount factor
TAU = 1e-3               # for soft update of target parameters
LR_ACTOR = 1e-5          # learning rate of the actor 
LR_CRITIC = 1e-5         # learning rate of the critic
WEIGHT_DECAY = 0         # L2 weight decay
NOISE_REDUCTION = 0.9999 # OU noise reduction

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Controller():
    """Interacts with and learns from the environment."""
    
    def __init__(self, state_size, action_size, random_seed):
        """Initialize a Controller object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            random_seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        # Actor Network (w/ Target Network)
        self.actor_local = Actor(state_size, action_size, random_seed).to(device)
        self.actor_target = Actor(state_size, action_size, random_seed).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        all_state_size = state_size * 2
        all_action_size = action_size * 2
        self.critic_local = Critic(all_state_size, all_action_size, random_seed).to(device)
        self.critic_target = Critic(all_state_size, all_action_size, random_seed).to(device)
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(), lr=LR_CRITIC, weight_decay=WEIGHT_DECAY)

        # Noise process
        self.noise = OUNoise(action_size, random_seed)
        self.noise_rate = 2
        
        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, random_seed)
    
    def save_experience(self, states, other_states, actions, other_actions, rewards, next_states, other_next_states, dones):
        """Save experience(s) in replay memory."""
        for state, other_state, action, other_action, reward, next_state, other_next_state, done in zip(states, other_states, actions, other_actions, rewards, next_states, other_next_states, dones):
            self.memory.add(state, other_state, action, other_action, reward, next_state, other_next_state, done)
        
    def act(self, state, add_noise=True):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state).cpu().data.numpy()
        self.actor_local.train()
        if add_noise:
            action += self.noise_rate * self.noise.sample()
            self.noise_rate *= NOISE_REDUCTION
        
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def learn(self, n_learning):
        # Learn, if enough samples are available in memory
        if len(self.memory) < BATCH_SIZE:
            return
        
        for _ in range(n_learning):
            self.update_models()
            
    def update_models(self):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        """
        states, other_states, actions, other_actions, rewards, next_states, other_next_states, dones = self.memory.sample()

        all_states = torch.cat((states, other_states), dim=1)
        all_actions = torch.cat((actions, other_actions), dim=1)
        all_next_states = torch.cat((next_states, other_next_states), dim=1)
        
        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        other_actions_next = self.actor_target(other_next_states)
        all_actions_next = torch.cat((actions_next, other_actions_next), dim=1)
        Q_targets_next = self.critic_target(all_next_states, all_actions_next)
        
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))
        
        # Compute critic loss
        Q_expected = self.critic_local(all_states, all_actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # ---------------------------- update actor ---------------------------- #
        # Compute actor loss
        actions_pred = self.actor_local(states)
        other_actions_pred = self.actor_local(other_states)
        all_actions_pred = torch.cat((actions_pred, other_actions_pred), dim=1)
        actor_loss = -self.critic_local(all_states, all_actions_pred).mean()
                                     
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # ----------------------- update target networks ----------------------- #
        self.soft_update_target_model(self.critic_local, self.critic_target, TAU)
        self.soft_update_target_model(self.actor_local, self.actor_target, TAU)
    
    def soft_update_target_model(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)