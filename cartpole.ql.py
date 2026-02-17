#!/usr/bin/env python3
"""CartPole-v1 cleaned up and adapted https://gymnasium.farama.org/introduction/train_agent/"""
from collections import defaultdict, namedtuple
from tqdm import tqdm  # Progress bar
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym

GAME_ID = "CartPole-v1"

# Training hyperparameters
LEARNING_RATE = 0.01        # How fast to learn (higher = faster but less stable)
N_EPISODES = 100_000        # Number of hands to practice
INITIAL_EPSILON = 1.0       # Start with 100% random actions
EPSILON_DECAY = INITIAL_EPSILON / (N_EPISODES / 2)  # Reduce exploration over time
FINAL_EPSILON = 0.1         # Always keep some exploration
DISCOUNT_FACTOR = 0.95
ROLLING_LENGTH = 500        #matplotlib Smooth over a 500-episode window

# types
HyperParams = namedtuple('HyperParams', [
'learning_rate', 'initial_epsilon', 'epsilon_decay', 'final_epsilon', 'discount_factor'])
Experience = namedtuple('Experience', ['obs', 'action', 'reward', 'terminated', 'next_obs'])

class BlackjackAgent:
    """Back jack agent"""
    def __init__(
        self,
        env_l: gym.Env,
        hyper_params: HyperParams,
    ):
        """Initialize a Q-Learning agent.

        Args:
            env: The training environment
            learning_rate: How quickly to update Q-values (0-1)
            initial_epsilon: Starting exploration rate (usually 1.0)
            epsilon_decay: How much to reduce epsilon each episode
            final_epsilon: Minimum exploration rate (usually 0.1)
            discount_factor: How much to value future rewards (0-1)
        """
        self.env = env_l

        # Q-table: maps (state, action) to expected reward
        # defaultdict automatically creates entries with zeros for new states
        self.q_values = defaultdict(lambda: np.zeros(self.env.action_space.n))

        # How much we care about future rewards
        # Exploration parameters
        self.hyper_params = hyper_params
        self.epsilon = hyper_params.initial_epsilon

        # Track learning progress
        self.training_error = []

    def get_action(self, obs_l: tuple[int, int, bool]) -> int:
        """Choose an action using epsilon-greedy strategy.

        Returns:
            action: 0 (stand) or 1 (hit)
        """
        # With probability epsilon: explore (random action)
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()

        # With probability (1-epsilon): exploit (best known action)

        return int(np.argmax(self.q_values[obs_l]))

    def update(
        self,
        experience : Experience
    ):
        """Update Q-value based on experience.

        This is the heart of Q-learning: learn from (state, action, reward, next_state)
        """
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
        self.hyper_params.final_epsilon, self.epsilon -
        self.hyper_params.epsilon_decay)


def get_moving_avgs(arr, window, convolution_mode):
    """Compute moving average to smooth noisy data."""
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window


def run():
    """Create environment and agent"""
    env_l = gym.make(GAME_ID)
    env_l = gym.wrappers.RecordEpisodeStatistics(env_l, buffer_length=N_EPISODES)

    agent_l = BlackjackAgent(
        env_l=env_l,
        hyper_params=HyperParams(
        LEARNING_RATE, INITIAL_EPSILON, EPSILON_DECAY, FINAL_EPSILON, DISCOUNT_FACTOR)
    )

    for _ in tqdm(range(N_EPISODES)):
        # Start a new hand
        obs, _ = env_l.reset()
        obs = tuple(obs)

        done = False

        # Play one complete hand
        while not done:
            # Agent chooses action (initially random, gradually more intelligent)
            action = agent_l.get_action(obs)

            # Take action and observe result
            next_obs, reward, terminated, truncated, _ = env_l.step(action)
            next_obs = tuple(next_obs)

            # Learn from this experience
            agent_l.update(Experience(obs, action, reward, terminated, next_obs))

            # Move to next state
            done = terminated or truncated
            obs = next_obs

        # Reduce exploration rate (agent becomes less random over time)
        agent_l.decay_epsilon()
    return env_l, agent_l

def plot(env_l, agent_l):
    """Smooth over a 500-episode window"""
    _, axs = plt.subplots(ncols=3, figsize=(12, 5))

    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env_l.return_queue,
        ROLLING_LENGTH,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")

    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env_l.length_queue,
        ROLLING_LENGTH,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")

    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent_l.training_error,
        ROLLING_LENGTH,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    env, agent = run()
    plot(env, agent)
