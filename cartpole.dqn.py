#!/usr/bin/env python3
"""CartPole-v1 cleaned up and converted to DQN
https://docs.pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://gymnasium.farama.org/introduction/train_agent/"""
# import argparse
from collections import namedtuple
# import pickle
import math
import random
import torch
from torch import nn
from torch import optim
# from tqdm import tqdm  # Progress bar
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
Experience = namedtuple('Experience', ['state', 'action', 'reward', 'terminated', 'next_state'])
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


class Agent:
    """Agent"""
    def __init__(self, game_id=GAME_ID):
        self.env = gym.make(game_id)
        self.env = gym.wrappers.RecordEpisodeStatistics(self.env, buffer_length=N_EPISODES)
        self.memory = []

    def train(self):
        """training"""
        episode_durations = []

        policy_net = DQN(self.env).to(DEVICE)
        target_net = DQN(self.env).to(DEVICE)
        target_net.load_state_dict(policy_net.state_dict())

        optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
        memory = []

        if torch.cuda.is_available() or torch.backends.mps.is_available():
            num_episodes = 600
            # num_episodes = 200
        else:
            num_episodes = 50

        steps_done = 0
        for _ in range(num_episodes):
            # Initialize the environment and get its state
            state, _ = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=DEVICE).unsqueeze(0)
            done = False
            t = 0
            while not done:
                t += 1
                action, steps_done = self.select_action(policy_net, state,  steps_done)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
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
                    optimizer, random.sample(memory, BATCH_SIZE), policy_net, target_net)

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = target_net.state_dict()
                policy_net_state_dict = policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key]*TAU + target_net_state_dict[key]*(1-TAU)
                target_net.load_state_dict(target_net_state_dict)

                if terminated or truncated:
                    episode_durations.append(t + 1)
                    done = True


        print('Complete')
        Plot(self.env, self).plot()

    def select_action(self, policy_net_l, state_l, steps):
        """Select action"""
        sample = random.random()
        eps_threshold = EPSILON_FINAL + (EPSILON_INITIAL - EPSILON_FINAL) * \
            math.exp(-1. * steps / EPSILON_DECAY)
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

    def optimize_model(self, optimizer_l, transitions, policy_net_l, target_net_l):
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



if __name__ == "__main__":

    # parser = argparse.ArgumentParser()
    # parser.add_argument("-f", help ='q-value filename')
    # args = parser.parse_args()
    # en = Env(args.f)
    # pl = Plot(en, en.learn())
    # pl.plot()
    Agent().train()
