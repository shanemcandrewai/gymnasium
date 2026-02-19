#!/usr/bin/env python3
"""CartPole-v1 cleaned up and convert to DQN
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://gymnasium.farama.org/introduction/train_agent/"""
import argparse
from collections import defaultdict, namedtuple
import pickle
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from tqdm import tqdm  # Progress bar
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym

GAME_ID = "CartPole-v1"

# Training hyperparameters
LEARNING_RATE = 0.01        # How fast to learn (higher = faster but less stable)
N_EPISODES = 10000         # Number of hands to practice
EPSILON_INITIAL = 1.0       # Start with 100% random actions
EPSILON_DECAY = EPSILON_INITIAL / (N_EPISODES / 2)  # Reduce exploration over time
EPSILON_FINAL = 0.1         # Always keep some exploration
TAU = 0.005
LR = 0.0003
DISCOUNT_FACTOR = 0.95
ROLLING_LENGTH = 500        #matplotlib Smooth over a 500-episode window

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
NUM_FEATURES = 128
BATCH_SIZE = 128
GAMMA = 0.99

# types
HyperParams = namedtuple('HyperParams', [
'learning_rate', 'epsilon_initial', 'epsilon_decay', 'epsilon_final', 'discount_factor'])
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'terminated', 'next_obs'])
QvalueInfo = namedtuple('QvalueInfo', [
'cart_pos_min', 'cart_vel_min', 'pole_ang_min', 'pole_ang_vel_min', 'cart_pos_max', \
'cart_vel_max', 'pole_ang_max', 'pole_ang_vel_max'])

class DQN(nn.Module):
    """Deep Q-network"""
    def __init__(self, env):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(np.prod(env.observation_space.shape), NUM_FEATURES),
            nn.ReLU(),
            nn.Linear(NUM_FEATURES, NUM_FEATURES),
            nn.ReLU(),
            nn.Linear(NUM_FEATURES, env.action_space.n)
        )
        self.memory = Experience()


    def forward(self, x):
        """forward pass"""
        logits = self.linear_relu_stack(x)
        return logits

    def optimize_model(self, optimizer_l, transitions, policy_net_l, target_net_l):
        """Optimize model"""
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Experience(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_obs)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_obs
                                                    if s is not None])
        state_batch = torch.cat(batch.obs)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net_l(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net_l(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer_l.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net_l.parameters(), 100)
        optimizer_l.step()



class CartPoleAgent:
    """Cart Pole agent"""
    def __init__(
        self,
        env: gym.Env,
        hyper_params: HyperParams,
        q_values_file=None
    ):
        """Initialize a Q-Learning agent.
        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            epsilon_initial: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            epsilon_final: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env

        # self.experience = Experience(obs_quant, action, reward, terminated, next_obs_quant)

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values_file = q_values_file
        # How much we care about future rewards
        # Exploration parameters
        self.hyper_params = hyper_params

        try:
            with open(q_values_file, 'rb') as f:
                self.q_values = defaultdict(
                lambda: np.zeros(self.env.action_space.n), pickle.load(f))
                self.epsilon = hyper_params.epsilon_final
        except (OSError, TypeError):
            self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))
            self.epsilon = hyper_params.epsilon_initial


        # Track learning progress
        self.training_error = []

        self.policy_net = DQN().to(DEVICE)
        self.target_net = DQN().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)



    def get_action(self, obs: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.
        Returns:
            0: Push cart to the left
            1: Push cart to the right
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)
        return self.policy_net(obs).argmax(0)

    def update(self, experience : Experience):
        """Update Q-value based on experience"""

        # This is the heart of Q-learning: learn from (state, action, reward, next_state)
        # What's the best we could do from the next state?
        # (Zero if episode terminated - no future rewards possible)
        future_q_value = (not experience.terminated) * np.max(self.q_values[experience.next_obs])

        # What should the Q-value be? (Bellman equation)
        target = experience.reward + self.hyper_params.discount_factor * future_q_value

        # How wrong was our current estimate?
        temporal_difference = target - self.q_values[experience.obs][experience.action]

        # Update our estimate in the direction of the error
        # Learning rate controls how big steps we take
        self.q_values[experience.obs][experience.action] = (
            self.q_values[experience.obs][
            experience.action] + self.hyper_params.learning_rate * temporal_difference
        )

        # Track learning progress (useful for debugging)
        self.training_error.append(temporal_difference)

    def decay_epsilon(self):
        """Reduce exploration rate after each episode."""
        self.epsilon = max(
        self.hyper_params.epsilon_final, self.epsilon -
        self.hyper_params.epsilon_decay)

    def get_info(self):
        """return Q-value stats"""
        qnp = np.array(list(self.q_values))
        qnp_min = qnp.min(0)
        qnp_max = qnp.max(0)

        try:
            with open(self.q_values_file, 'wb') as f:
                pickle.dump(dict(self.q_values), f, pickle.HIGHEST_PROTOCOL)
        except (OSError, TypeError):
            pass

        return QvalueInfo(qnp_min[0], qnp_min[1], qnp_min[2], qnp_min[3], qnp_max[
        0], qnp_max[1], qnp_max[2], qnp_max[3])



class Env:
    """Training environment"""
    def __init__(self, q_values_file=None):
        """Initialise gym environment"""
        self.env = gym.make(GAME_ID, render_mode="human")
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=N_EPISODES)

        self.agent = CartPoleAgent(
            self.env,
            hyper_params=HyperParams(
            LEARNING_RATE, EPSILON_INITIAL, EPSILON_DECAY, EPSILON_FINAL, DISCOUNT_FACTOR),
            q_values_file=q_values_file
        )

    def calcuate_bins(self):
        """Calculate bins for action spaces"""
        # cart_pos_min = env.observation_space.low[0]
        cart_pos_min = -2.4
        cart_vel_min = -3.8
        # pole_ang_min = env.observation_space.low[2]
        pole_ang_min = -.2095
        pole_ang_vel_min = -3.4
        # cart_pos_max = env.observation_space.high[0]
        cart_pos_max = 2.4
        cart_vel_max = 3.8
        # pole_ang_max = env.observation_space.high[2]
        pole_ang_max = .2095
        pole_ang_vel_max = 3.4

        num_bins = 7

        return np.linspace(cart_pos_min, cart_pos_max, num_bins), np.linspace(
        cart_vel_min, cart_vel_max, num_bins), np.linspace(
        pole_ang_min, pole_ang_max, num_bins), np.linspace(
        pole_ang_vel_min, pole_ang_vel_max, num_bins)

    def quantize(self, obs, obs_bins):
        """Quantize observation into action space bins"""
        return np.digitize(obs[0], obs_bins[0]), np.digitize(
        obs[1], obs_bins[1]), np.digitize(obs[2], obs_bins[2]),np.digitize(
        obs[3], obs_bins[3])


    def learn(self):
        """Create agent, start learning"""

        obs_bins = self.calcuate_bins()
        for _ in tqdm(range(N_EPISODES)):
            # Start a new hand
            obs, _ = self.env.reset()
            obs_quant = self.quantize(obs, obs_bins)
            done = False

            # Play one complete hand
            while not done:
                # Agent chooses action (initially random, gradually more intelligent)
                action = self.agent.get_action(obs_quant)

                # Take action and observe result
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                next_obs_quant = self.quantize(next_obs, obs_bins)

                # Learn from this experience
                self.agent.update(Experience(obs_quant, action, reward, terminated, next_obs_quant))

                # Move to next state
                done = terminated or truncated
                obs_quant = next_obs_quant

            # Reduce exploration rate (agent becomes less random over time)
            self.agent.decay_epsilon()
        print(self.agent.get_info())
        return self.agent.training_error

class Plot:
    """Results plotter"""
    def __init__(self, env, training_error):
        self.env = env
        self.training_error = training_error

    def get_moving_avgs(self, arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    def plot(self):
        """Smooth over a 500-episode window"""
        _, axs = plt.subplots(ncols=3, figsize=(12, 5))

        # Episode rewards (win/loss performance)
        axs[0].set_title("Episode rewards")
        reward_moving_average = self.get_moving_avgs(
            self.env.return_queue,
            ROLLING_LENGTH,
            "valid"
        )
        axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
        axs[0].set_ylabel("Average Reward")
        axs[0].set_xlabel("Episode")

        # Episode lengths (how many actions per hand)
        axs[1].set_title("Episode lengths")
        length_moving_average = self.get_moving_avgs(
            self.env.length_queue,
            ROLLING_LENGTH,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Length")
        axs[1].set_xlabel("Episode")

        # Training error (how much we're still learning)
        axs[2].set_title("Training Error")
        training_error_moving_average = self.get_moving_avgs(
            self.training_error,
            ROLLING_LENGTH,
            "same"
        )
        axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
        axs[2].set_ylabel("Temporal Difference Error")
        axs[2].set_xlabel("Step")

        plt.tight_layout()
        plt.show()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help ='q-value filename')
    args = parser.parse_args()
    en = Env(args.f)
    pl = Plot(en, en.learn())
    pl.plot()
