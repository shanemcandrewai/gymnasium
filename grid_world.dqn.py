#!/usr/bin/env python3
"""cartpole.dqn.py adapted to gridworld.gymnasium_env:GridWorld-v0
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://gymnasium.farama.org/introduction/train_agent/"""
import argparse
from collections import namedtuple
import math
import random
import torch
from torch import nn
from torch import optim
from tqdm import tqdm  # Progress bar
from matplotlib import pyplot as plt
import numpy as np
import gymnasium as gym

GAME_ID = "gridworld.gymnasium_env:GridWorld-v0"

# Training hyperparameters
EPSILON_INITIAL = 0.9       # Start with 100% random actions
EPSILON_FINAL = 0.05         # Always keep some exploration
TAU = 0.005
LEARNING_RATE = 0.0003
DISCOUNT_FACTOR = 0.95
ROLLING_LENGTH = 10        #matplotlib Smooth over a 500-episode window

DEVICE = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
NUM_FEATURES = 128
BATCH_SIZE = 128
GAMMA = 0.99

# types
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'terminated', 'next_state'])

class DQN(nn.Module):
    """Deep Q-network"""

    def __init__(self, env):
        super().__init__()
        observation_space_num = np.prod(list(y for y in zip(*(
        env.observation_space[x].shape for x in env.observation_space))))
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(observation_space_num, NUM_FEATURES),
            nn.ReLU(),
            nn.Linear(NUM_FEATURES, NUM_FEATURES),
            nn.ReLU(),
            nn.Linear(NUM_FEATURES, env.action_space.n)
        )

    def forward(self, x):
        """forward pass"""
        logits = self.linear_relu_stack(x)
        return logits

class Agent:
    """Agent"""
    params = {}

    def __init__(self, model_file=None, game_id=GAME_ID):
        if game_id is None:
            self.env = gym.make(GAME_ID, render_mode="human")
            # self.env = gym.make(GAME_ID)
        else:
            self.env = gym.make(game_id, render_mode="human")
            # self.env = gym.make(game_id)
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            self.num_episodes = 600
        else:
            # self.num_episodes = 50
            self.num_episodes = 600
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=self.num_episodes)
        self.model_file = model_file
        self.policy_net = DQN(self.env).to(DEVICE)
        try:
            self.policy_net.load_state_dict(torch.load(model_file, weights_only=True))
            self.params['epsilon_initial'] = EPSILON_FINAL  # Start with fewer random actions
        except (OSError, TypeError, AttributeError):
            self.params['epsilon_initial'] = EPSILON_INITIAL # Start with 100% random actions

        self.params['epsilon_decay'] = self.num_episodes
        self.target_net = DQN(self.env).to(DEVICE)
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LEARNING_RATE, amsgrad=True)

    def train(self):
        """training"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
        memory = []
        steps_done = 0
        episode = 0
        for episode in tqdm(range(self.num_episodes)):
            # Initialize the environment and get its state
            state, _ = self.env.reset()
            state = np.concatenate([state[x] for x in state], dtype=np.float32)
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action, steps_done = self.select_action(
                self.policy_net, state,  steps_done, episode)
                obs, reward, terminated, truncated, _ = self.env.step(action)

                if terminated:
                    next_state = None
                else:
                    next_state = np.concatenate([obs[x] for x in obs], dtype=np.float32)

                # Store the transition in memory
                memory.append(Experience(state, action, reward, terminated, next_state))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(memory) >= BATCH_SIZE:
                    self.optimize_model(
                    self.optimizer, random.sample(
                    memory, BATCH_SIZE), self.policy_net, self.target_net)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)

        print('Complete')
        self.env.close()
        if episode > 1:
            try:
                torch.save(self.policy_net.state_dict(), self.model_file)
            except (OSError, TypeError, AttributeError):
                pass
            Plot(self.env).plot()

    def select_action(self, policy_net, state, steps, episode):
        """Select action"""
        steps += 1
        action = -1
        if self.params['epsilon_initial'] == EPSILON_FINAL:
            eps_threshold = EPSILON_FINAL
        else:
            decay_rate = math.exp(-2 * episode / self.num_episodes)
            eps_threshold = np.clip(decay_rate, EPSILON_FINAL, self.params['epsilon_initial'])
        if random.random() > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max resul t is index of where max element was
                # found, so we pick action with the larger expected reward.
                logits = policy_net(torch.from_numpy(state).to(DEVICE))
                action = np.int64(logits.argmax().cpu())
        else:
            action = self.env.action_space.sample()
        return action, steps

    def optimize_model(self, optimizer, transitions, policy_net, target_net):
        """Optimize model"""
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Experience(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = np.array([isinstance(s, np.ndarray) for s in batch.next_state])
        non_final_next_states = torch.from_numpy(np.array([
        s for s in batch.next_state if s is not None])).to(DEVICE)

        state_batch = torch.from_numpy(np.array(batch.state)).to(DEVICE)
        action_batch = torch.from_numpy(np.array(list(batch.action))).to(DEVICE)
        reward_batch = torch.from_numpy(np.array(batch.reward)).to(DEVICE)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = policy_net(state_batch).gather(1, action_batch.unsqueeze(1))

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE).to(DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = nn.SmoothL1Loss()(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
        optimizer.step()

class Plot:
    """Results plotter"""
    def __init__(self, env):
        self.env = env

    def get_moving_avgs(self, arr, window, convolution_mode):
        """Compute moving average to smooth noisy data."""
        return np.convolve(
            np.array(arr).flatten(),
            np.ones(window),
            mode=convolution_mode
        ) / window

    def plot(self):
        """Smooth over a 500-episode window"""
        _, axs = plt.subplots(ncols=2, figsize=(12, 5))

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

        # Episode times
        axs[1].set_title("Episode times")
        length_moving_average = self.get_moving_avgs(
            self.env.time_queue,
            ROLLING_LENGTH,
            "valid"
        )
        axs[1].plot(range(len(length_moving_average)), length_moving_average)
        axs[1].set_ylabel("Average Episode Time")
        axs[1].set_xlabel("Episode")

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", help ='q-value filename')
    parser.add_argument("-g", help ='Game ID')
    args = parser.parse_args()
    Agent(args.f, args.g).train()
