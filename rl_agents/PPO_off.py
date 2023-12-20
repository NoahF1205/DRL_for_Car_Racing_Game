import torch.nn as nn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Beta
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from env.car_racing_wrapper import Env
from rl_agents.rl_utils import *


class PolicyNet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, input_channels, hidden_dim):
        super(PolicyNet, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU(), nn.Linear(100, 1))
        self.fc = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU())
        self.alpha_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Softplus())
        self.beta_head = nn.Sequential(nn.Linear(hidden_dim, 3), nn.Softplus())
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = x.to(torch.float)
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        x = self.fc(x)
        alpha = self.alpha_head(x) + 1
        beta = self.beta_head(x) + 1
        return alpha, beta

class ValueNet(nn.Module):
    """
    Actor-Critic Network for PPO
    """

    def __init__(self, input_channels, hidden_dim):
        super(ValueNet, self).__init__()
        self.cnn_base = nn.Sequential(  # input shape (4, 96, 96)
            nn.Conv2d(input_channels, 8, kernel_size=4, stride=2),
            nn.ReLU(),  # activation
            nn.Conv2d(8, 16, kernel_size=3, stride=2),  # (8, 47, 47)
            nn.ReLU(),  # activation
            nn.Conv2d(16, 32, kernel_size=3, stride=2),  # (16, 23, 23)
            nn.ReLU(),  # activation
            nn.Conv2d(32, 64, kernel_size=3, stride=2),  # (32, 11, 11)
            nn.ReLU(),  # activation
            nn.Conv2d(64, 128, kernel_size=3, stride=1),  # (64, 5, 5)
            nn.ReLU(),  # activation
            nn.Conv2d(128, 256, kernel_size=3, stride=1),  # (128, 3, 3)
            nn.ReLU(),  # activation
        )  # output shape (256, 1, 1)
        self.v = nn.Sequential(nn.Linear(256, hidden_dim), nn.ReLU(), nn.Linear(100, 1))
        self.apply(self._weights_init)

    @staticmethod
    def _weights_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('relu'))
            nn.init.constant_(m.bias, 0.1)

    def forward(self, x):
        x = x.to(torch.float)
        x = self.cnn_base(x)
        x = x.view(-1, 256)
        v = self.v(x)
        return v


class PPO_off():
    """
    Agent for training
    """
    def __init__(self, input_channel, hidden_dim, gamma, eps, ppo_epoch, buffer_capacity, batch_size, actor_lr, critic_lr):
        self.max_grad_norm = 0.5
        self.clip_param = eps
        self.training_step = 0
        self.ppo_epoch = ppo_epoch
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.input_channel = input_channel
        self.hidden_dim = hidden_dim
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor_net = PolicyNet(self.input_channel, self.hidden_dim).to(self.device)
        self.critic_net = ValueNet(self.input_channel, self.hidden_dim).to(self.device)
        self.transition = np.dtype(
            [('s', np.float64, (input_channel, 96, 96)), ('a', np.float64, (3,)), ('a_logp', np.float64),
             ('r', np.float64), ('s_', np.float64, (input_channel, 96, 96))])
        self.buffer = np.empty(self.buffer_capacity, dtype=self.transition)
        self.counter = 0
        self.actor_optimizer = optim.Adam(self.actor_net.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(self.critic_net.parameters(), lr=critic_lr)
        self.gamma = gamma

    def select_action(self, state):
        state = torch.from_numpy(state).double().to(self.device).unsqueeze(0)
        # state:4*96*96 --> 1*4*96*96
        with torch.no_grad():
            alpha, beta = self.actor_net(state)
        dist = Beta(alpha, beta)
        action = dist.sample()
        a_logp = dist.log_prob(action).sum(dim=1)

        action = action.squeeze().cpu().numpy()
        a_logp = a_logp.item()
        return action, a_logp

    def store(self, transition):
        self.buffer[self.counter] = transition
        self.counter += 1
        if self.counter == self.buffer_capacity:
            self.counter = 0
            return True
        else:
            return False

    def update(self):
        self.training_step += 1
        s = torch.tensor(self.buffer['s'], dtype=torch.double).to(self.device)
        a = torch.tensor(self.buffer['a'], dtype=torch.double).to(self.device)
        r = torch.tensor(self.buffer['r'], dtype=torch.double).to(self.device).view(-1, 1)
        s_ = torch.tensor(self.buffer['s_'], dtype=torch.double).to(self.device)
        old_a_logp = torch.tensor(self.buffer['a_logp'], dtype=torch.double).to(self.device).view(-1, 1)

        with torch.no_grad():
            target_v = r + self.gamma * self.critic_net(s_)
            adv = target_v - self.critic_net(s)


        for _ in range(self.ppo_epoch):
            for index in BatchSampler(SubsetRandomSampler(range(self.buffer_capacity)), self.batch_size, False):
                alpha, beta = self.actor_net(s[index])
                dist = Beta(alpha, beta)
                a_logp = dist.log_prob(a[index]).sum(dim=1, keepdim=True)
                ratio = torch.exp(a_logp - old_a_logp[index])
                surr1 = ratio * adv[index]
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv[index]
                action_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.smooth_l1_loss(self.critic_net(s[index]), target_v[index])
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                action_loss.backward()
                value_loss.backward()
                self.actor_optimizer.step()
                self.critic_optimizer.step()

    def load_weights(self, actor_path, critic_path):
        self.actor_net.load_state_dict(torch.load(actor_path))
        self.critic_net.load_state_dict(torch.load(critic_path))



def train_car_racing(agent, env, num_episodes):
    start_time = time.time()
    return_list = []
    for i_episode in range(num_episodes):
        episode_return = 0
        state = env.reset()
        while True:
            action, a_logp = agent.select_action(state)
            next_state, reward, done, die = env.step(action * np.array([2., 1., 1.]) + np.array([-1., 0., 0.]))
            if agent.store((state, action, a_logp, reward, next_state)):
                print('updating')
                agent.update()
            episode_return += reward
            state = next_state
            if done or die:
                break
        return_list.append(episode_return)

        print("epoch: ", i_episode, " episode_return: ", episode_return)
    env.close()
    training_time = time.time() - start_time
    print("Training finished. Total time: {:.2f}s".format(training_time))
    torch.save(agent.actor_net.state_dict(), './weights/PPO_off_actor_weights.pth')
    torch.save(agent.critic_net.state_dict(), './weights/PPO_off_critic_weights.pth')

#Main training loop
if __name__ == "__main__":
    # env = Env(action_repeat=8, img_stack=4, seed=0, render_mode=None)
    # agent = PPO_off(input_channel=4, hidden_dim=100, gamma=0.99, eps=0.1, ppo_epoch=10, buffer_capacity=200, batch_size=128, actor_lr=1e-3,
    #               critic_lr=1e-3)
    # train_off_policy_agent_off(agent, env, 3)
    actor_lr = 1e-3
    critic_lr = 1e-3
    num_episodes = 100
    hidden_dim = 100
    gamma = 0.99
    ppo_epoch = 10
    eps = 0.1
    buffer_capacity = 2000
    batch_size = 128

    env = Env(seed=0, action_repeat=8, img_stack=3, render_mode=None)
    torch.manual_seed(0)

    input_channels = 3

    agent = PPO_off(input_channel=input_channels, hidden_dim=hidden_dim, gamma=gamma, eps=eps, ppo_epoch=ppo_epoch,
                    buffer_capacity=buffer_capacity, batch_size=batch_size, actor_lr=actor_lr,
                    critic_lr=critic_lr)
    return_list, training_time = train_off_policy_agent_off(agent, env, num_episodes)


