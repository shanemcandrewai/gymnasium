#!/usr/bin/env python3
"""CartPole-v1 cleaned up and converted to DQN
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
EPSILON_FINAL = 0.01         # Always keep some exploration
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
        else:
            self.env = gym.make(game_id, render_mode="human")
        # self.env = gym.make(game_id)
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            self.num_episodes = 600
        else:
            # self.num_episodes = 50
            self.num_episodes = 200
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=self.num_episodes)
        self.model_file = model_file
        self.policy_net = DQN(self.env).to(DEVICE)
        try:
            self.policy_net.load_state_dict(torch.load(model_file, weights_only=True))
            self.params['epsilon_initial'] = EPSILON_FINAL  # Start with fewer random actions
        except (OSError, TypeError, AttributeError):
            self.params['epsilon_initial'] = EPSILON_INITIAL # Start with 100% random actions

        self.params['epsilon_decay'] = self.params['epsilon_initial'] / (
        self.num_episodes / 2)  # Reduce exploration over time
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
            state, info = self.env.reset()
            state = torch.tensor(np.concatenate(list(
            state[x] for x in state)), dtype=torch.float32, device=DEVICE).unsqueeze(0)
            terminated = False
            truncated = False
            while not terminated and not truncated:
                action, steps_done = self.select_action(self.policy_net, state,  steps_done)
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                observation = torch.tensor(np.concatenate(list(observation[
                x] for x in observation)), dtype=torch.float32, device=DEVICE)
                reward = torch.tensor([reward], device=DEVICE)

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                    observation, dtype=torch.float32, device=DEVICE).unsqueeze(0)

                # Store the transition in memory
                memory.append(Experience(state, action, reward, terminated, next_state))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                if len(memory) >= BATCH_SIZE:
                    self.optimize_model(
                    self.optimizer, random.sample(
                    memory, BATCH_SIZE))

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key]*TAU + target_net_state_dict[key]*(1-TAU)
                self.target_net.load_state_dict(target_net_state_dict)
            if info['episode']['l'] >= 500:
                break

        print('Complete')
        self.env.close()
        if episode > 1:
            try:
                torch.save(self.policy_net.state_dict(), self.model_file)
            except (OSError, TypeError, AttributeError):
                pass
            Plot(self.env).plot()

    def select_action(self, policy_net_l, state_l, steps):
        """Select action"""
        sample = random.random()
        eps_threshold = EPSILON_FINAL + (self.params['epsilon_initial'] - EPSILON_FINAL) * \
-            math.exp(-1. * steps / self.params['epsilon_decay'])
        steps += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max resul t is index of where max element was
                # found, so we pick action with the larger expected reward.
                return policy_net_l(state_l).max(1).indices.view(1, 1), steps
        else:
            return torch.tensor([[
            self.env.action_space.sample()]], device=DEVICE, dtype=torch.long), steps

    def optimize_model(self, optimizer_l, transitions):
        """Optimize model"""
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Experience(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=DEVICE, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer_l.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        optimizer_l.step()

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
