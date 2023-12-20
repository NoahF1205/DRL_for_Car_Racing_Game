import os
import torch.nn as nn
import torch.nn.functional as F
import rl_agents.rl_utils as rl_utils
from rl_agents.rl_utils import *

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


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


class PPO_on:
    def __init__(self, input_channels, hidden_dim, action_dim, actor_lr, critic_lr, gamma, lmbda, ppo_epoch, eps):
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
        self.lmbda = lmbda
        self.epochs = ppo_epoch  # 一条序列的数据用来训练轮数
        self.eps = eps  # PPO中截断范围的参数

    def take_action(self, state):
        state = torch.tensor(np.array([state]), dtype=torch.float).to(self.device)
        # Get the probability distribution over actions
        with torch.no_grad():
            probs = self.actor(state)
            # print("probs", probs)
        # print(probs)
        action_dist = torch.distributions.Categorical(probs)
        action_index = action_dist.sample().item()  # This is an index, not the actual action
        # print("action_index:", action_index)
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
        advantage = rl_utils.compute_advantage(self.gamma, self.lmbda,
                                               td_delta.cpu()).to(self.device)
        old_log_probs = torch.log(self.actor(states).gather(1,
                                                            actions)).detach()

        for _ in range(self.epochs):
            log_probs = torch.log(self.actor(states).gather(1, actions))
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps,
                                1 + self.eps) * advantage  # 截断
            actor_loss = torch.mean(-torch.min(surr1, surr2))  # PPO损失函数
            critic_loss = torch.mean(
                F.mse_loss(self.critic(states), td_target.detach()))
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

    def load_weights(self, actor_path, critic_path):
        self.actor.load_state_dict(torch.load(actor_path))
        self.critic.load_state_dict(torch.load(critic_path))

# actor_lr = 1e-3
# critic_lr = 1e-3
# num_episodes = 100
# hidden_dim = 128
# gamma = 0.98
# lmbda = 0.95
# ppo_epoch = 10
# eps = 0.2
#
# env = Env(seed=0, action_repeat=8, img_stack=3,render_mode=None)
# torch.manual_seed(0)
#
#
# input_channels = 3
# action_dim = 10
#
# agent = PPO_on(input_channels, hidden_dim, action_dim, actor_lr, critic_lr, gamma,lmbda,ppo_epoch,eps)
# return_list, training_time = train_on_policy_agent_on(env, agent, num_episodes)
