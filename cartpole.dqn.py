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

    def forward(self, x):
        """forward pass"""
        logits = self.linear_relu_stack(x)
        return logits


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
        return int(np.argmax(self.q_values[obs]))

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

def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window

def init():
    """Initialise gym environment"""
    env = gym.make(GAME_ID, render_mode="human")
    env = gym.wrappers.RecordEpisodeStatistics(env, buffer_length=N_EPISODES)
    return env

def calcuate_bins():
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

def quantize(obs, obs_bins):
    """Quantize observation into action space bins"""
    return np.digitize(obs[0], obs_bins[0]), np.digitize(
    obs[1], obs_bins[1]), np.digitize(obs[2], obs_bins[2]),np.digitize(
    obs[3], obs_bins[3])

def learn(env, q_values_file=None):
    """Create agent, start learning"""
    agent = CartPoleAgent(
        env=env,
        hyper_params=HyperParams(
        LEARNING_RATE, EPSILON_INITIAL, EPSILON_DECAY, EPSILON_FINAL, DISCOUNT_FACTOR),
        q_values_file=q_values_file
    )

    obs_bins = calcuate_bins()
    for _ in tqdm(range(N_EPISODES)):
        # Start a new hand
        obs, _ = env.reset()
        obs_quant = quantize(obs, obs_bins)
        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent.get_action(obs_quant)

            # Take action and observe result
            next_obs, reward, terminated, truncated, _ = env.step(action)
            next_obs_quant = quantize(next_obs, obs_bins)

            # Learn from this experience
            agent.update(Experience(obs_quant, action, reward, terminated, next_obs_quant))

            # Move to next state
            done = terminated or truncated
            obs_quant = next_obs_quant

        # Reduce exploration rate (agent becomes less random over time)
        agent.decay_epsilon()
    print(agent.get_info())
    return env, agent

def plot(env, agent):
    """Smooth over a 500-episode window"""
    _, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        ROLLING_LENGTH,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        ROLLING_LENGTH,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
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
    plot(*learn(init(), args.f))
