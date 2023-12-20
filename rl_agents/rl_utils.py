import collections
import random
import time
import numpy as np
import torch
from tqdm import tqdm


class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), done

    def size(self):
        return len(self.buffer)


def moving_average(a, window_size):
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))


def transfer_action_shape(action_index):
    # Initialize an array for the action with zeros
    action = np.zeros(3)

    # Define soft and hard action values
    turn_left_soft, turn_left_hard = -0.5, -1.0
    turn_right_soft, turn_right_hard = 0.5, 1.0
    accelerate_soft, accelerate_hard = 0.5, 1.0
    brake_soft, brake_hard = 0.5, 0.8

    # Map the index to a specific action
    action_mappings = {
        0: [0, accelerate_hard, 0],  # Accelerate hard
        1: [0, accelerate_soft, 0],  # Accelerate soft
        2: [turn_left_hard, accelerate_hard, 0],  # Turn left hard & Accelerate hard
        3: [turn_left_hard, accelerate_soft, 0],  # Turn left soft & Accelerate soft
        4: [turn_left_soft, 0, brake_hard],  # Turn left hard & Brake hard
        5: [turn_left_soft, 0, brake_soft],  # Turn left soft & Brake soft
        6: [turn_right_soft, accelerate_hard, 0],  # Turn left hard & Accelerate hard
        7: [turn_right_hard, accelerate_soft, 0],  # Turn left soft & Accelerate soft
        8: [turn_right_soft, 0, brake_hard],  # Turn left hard & Brake hard
        9: [turn_right_hard, 0, brake_soft],  # Turn left soft & Brake soft
    }

    # Get the action from the mappings
    action = action_mappings.get(action_index, [0, 0, 0])

    # Make sure action is a list or array of floats
    return action


def train_on_policy_agent(env, agent, num_episodes):
    start_time = time.time()
    return_list = []
    for i_episode in tqdm(range(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            # print(type(state))
            action = agent.take_action(state)
            next_state, reward, terminated, truncated = env.step(transfer_action_shape(action))
            done = terminated or truncated
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward

        return_list.append(episode_return)
        agent.update(transition_dict)
        if (i_episode + 1) % 10 == 0:
            tqdm.write('Episode: {}, Return: {:.3f}, Average Return: {:.3f}'.format(
                i_episode + 1, episode_return, np.mean(return_list[-10:])))
    env.close()  # Close the environment when training is finished
    training_time = time.time() - start_time
    print("Training finished. Total time: {:.2f}s".format(training_time))
    torch.save(agent.actor.state_dict(), './weights/actor_weights.pth')
    torch.save(agent.critic.state_dict(), './weights/critic_weights.pth')

    return return_list, training_time


def train_off_policy_agent(env, agent, num_episodes, replay_buffer, minimal_size, batch_size):
    start_time = time.time()
    return_list = []
    # Start the game
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state = env.reset()
                # for i in range(24):
                #     env.env.step([0.0, 0.0, 0.0])
                # print('train_off_policy_agent :::: state : ', state.shape)
                done = False
                while not done:
                    # env.render()
                    action = agent.take_action(state)
                    # print('action : ', action)
                    next_state, reward, terminated, truncated = env.step(transfer_action_shape(action))
                    done = terminated or truncated
                    replay_buffer.add(state, action, reward, next_state, done)
                    state = next_state
                    episode_return += reward
                    # start training when buffer size > minimum
                    if replay_buffer.size() > minimal_size:
                        b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                        transition_dict = {'states': b_s, 'actions': b_a, 'next_states': b_ns, 'rewards': b_r,
                                           'dones': b_d}
                        agent.update(transition_dict)
                print(episode_return)
                return_list.append(episode_return)
                if (i_episode + 1) % 10 == 0:
                    pbar.set_postfix({'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                                      'return': '%.3f' % np.mean(return_list[-10:])})
                pbar.update(1)
    env.close()  # Close the environment when training is finished
    training_time = time.time() - start_time
    print("Training finished. Total time: {:.2f}s".format(training_time))
    torch.save(agent.q_net.state_dict(), './weights/DQN_weights.pth')
    return return_list, training_time


def play_game(env, agent, episodes=1):
    for _ in range(episodes):
        state = env.reset()
        done = False
        while not done:
            env.render()
            action = agent.take_action_greedy(state)
            next_state, reward, terminated, truncated = env.step(transfer_action_shape(action))
            done = terminated or truncated
    env.close()


def compute_advantage(gamma, lmbda, td_delta):
    td_delta = td_delta.detach().numpy()
    advantage_list = []
    advantage = 0.0
    for delta in td_delta[::-1]:
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)


def train_on_policy_agent_on(env, agent, num_episodes):
    start_time = time.time()
    return_list = []
    for i_episode in tqdm(range(num_episodes)):
        episode_return = 0
        transition_dict = {'states': [], 'actions': [], 'next_states': [], 'rewards': [], 'dones': []}
        state = env.reset()
        done = False
        while not done:
            # print(type(state))
            action = agent.take_action(state)
            next_state, reward, terminated, truncated = env.step(transfer_action_shape(action))
            done = terminated or truncated
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            state = next_state
            episode_return += reward

        return_list.append(episode_return)
        agent.update(transition_dict)
        if (i_episode + 1) % 10 == 0:
            tqdm.write('Episode: {}, Return: {:.3f}, Average Return: {:.3f}'.format(
                i_episode + 1, episode_return, np.mean(return_list[-10:])))
    env.close()  # Close the environment when training is finished
    training_time = time.time() - start_time
    print("Training finished. Total time: {:.2f}s".format(training_time))
    torch.save(agent.actor_net.state_dict(), './weights/PPO_on_actor_weights.pth')
    torch.save(agent.critic_net.state_dict(), './weights/PPO_on_critic_weights.pth')
    return return_list, training_time


def train_off_policy_agent_off(agent, env, num_episodes):
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
    return return_list, training_time
