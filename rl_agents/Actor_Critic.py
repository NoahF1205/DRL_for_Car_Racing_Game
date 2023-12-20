import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch.distributions import Beta

from rl_agents.rl_utils import *

class PolicyNet(nn.Module):
    def __init__(self, input_channels, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        # Convolutional base
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        # Fully connected layers and output head for value estimation
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        # Initialize weights
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        x = self.fc(x)
        return F.softmax(x, dim=1)


class ValueNet(nn.Module):
    def __init__(self, input_channels, hidden_dim):
        super(ValueNet, self).__init__()
        # Convolutional base
        self.cnn_base = nn.Sequential(
            nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        # Fully connected layers and output head for value estimation
        self.fc = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self.apply(self._weights_init)

    def _weights_init(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = self.cnn_base(x)
        x = x.view(x.size(0), -1)  # Flatten the output of conv layers
        return self.fc(x)

class ActorCritic:
    def __init__(self, input_channels, hidden_dim, action_dim, actor_lr, critic_lr, gamma):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Policy network
        self.actor = PolicyNet(input_channels, hidden_dim, action_dim).to(self.device)
        # Value network
        self.critic = ValueNet(input_channels, hidden_dim).to(self.device)

        # Optimizer for the policy network
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=critic_lr)  # Optimizer for the value network
        self.gamma = gamma

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # Get the probability distribution over actions
        with torch.no_grad():
            probs = self.actor(state)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs)
        action_index = action_dist.sample().item()  # This is an index, not the actual action
        return action_index

    def update(self, transition_dict):
        # Convert lists to tensors
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(np.array(transition_dict['next_states']), dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).to(self.device)

        # Check that the number of states matches the number of actions
        assert states.shape[0] == actions.shape[0], "The number of states must match the number of actions."

        # Reshape to match the expected shape for gather
        actions = actions.view(-1, 1)
        rewards = rewards.view(-1, 1)
        dones = dones.view(-1, 1)

        # Compute td_target and td_delta
        td_target = rewards + self.gamma * self.critic(next_states) * (1 - dones)
        td_delta = td_target - self.critic(states)
        # Get log probabilities
        probs = self.actor(states)
        log_probs = torch.log(probs.gather(1, actions))
        # Compute entropy
        entropy = -(probs * log_probs).sum(1, keepdim=True)

        # Compute losses
        actor_loss = torch.mean(-log_probs * td_delta.detach() - 0.05 * entropy)
        # actor_loss = torch.mean(-log_probs * td_delta.detach())
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))

        # Zero gradients
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()

        # Perform backpropagation
        actor_loss.mean().backward()
        critic_loss.mean().backward()

        # Update parameters
        self.actor_optimizer.step()
        self.critic_optimizer.step()

    def load_weights(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))
