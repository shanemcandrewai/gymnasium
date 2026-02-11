"""dql_cartpole"""
# Import necessary libraries
import random
from collections import deque
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gymnasium as gym

SEED=170715
# Define training parameters
NUM_EPISODES = 250
# NUM_EPISODES = 20
MAX_STEPS_PER_EPISODE = 200
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY_RATE = 0.99
GAMMA = 0.9
LR = 0.0025
BUFFER_SIZE = 10000
BATCH_SIZE = 128
UPDATE_FREQUENCY = 10
FC1_UNITS = 64
FC2_UNITS = 64
TAU = 0.001
GAME_ID = "CartPole-v1"
RENDER_GAME = "human"

# Check if GPU is available
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print(device)

# Define the neural network model
class QNetwork(nn.Module):
    """QNetwork"""
    def __init__(self, state_size, action_size, fc1_units=FC1_UNITS, fc2_units=FC2_UNITS):
        super().__init__()
        self.seed = torch.manual_seed(SEED)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.to(device)

    def forward(self, state_l):
        """ forward pass"""
        x = F.relu(self.fc1(state_l))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Define the DQN agent class
class DQNAgent:
    """DQN Agent"""
    # Initialize the DQN agent
    def __init__(self, state_size, action_size, lr):
        self.state_size = state_size
        self.action_size = action_size
        random.seed(SEED)

        self.qnetwork_local = QNetwork(state_size, action_size).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr)

        self.t_step = 0


    def choose_action(self, state_l, eps=0.):
        """Choose an action based on the current state"""
        state_tensor = torch.from_numpy(state_l).float().unsqueeze(0).to(device)

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state_tensor)
        self.qnetwork_local.train()

        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        return np.random.randint(self.action_size)

    def learn(self, experiences, gamma=GAMMA):
        """Learn from batch of experiences"""
        states, actions, rewards, next_states, dones = zip(*experiences)
        states = torch.from_numpy(np.vstack(states)).float().to(device)
        actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        q_expected = self.qnetwork_local(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.qnetwork_local, self.qnetwork_target)

    def soft_update(self, local_model, target_model, tau=TAU):
        """soft update"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


# Set up the environment
env = gym.make("CartPole-v1", MAX_STEPS_PER_EPISODE)

# Initialize the DQNAgent
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
new_agent = DQNAgent(input_dim, output_dim, lr = LR)

buffer = deque(maxlen=BUFFER_SIZE)


# Training loop
for episode in range(NUM_EPISODES):
    # Reset the environment
    state = env.reset()[0]
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY_RATE ** episode))

    step = 0
    done = False
    while not done:
        # Choose and perform an action
        action = new_agent.choose_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)

        buffer.append((state, action, reward, next_state, done))

        if len(buffer) >= BATCH_SIZE:
            batch = random.sample(buffer, BATCH_SIZE)
            # Update the agent's knowledge
            new_agent.learn(batch)

        state = next_state
        step += 1

    if (episode + 1) % UPDATE_FREQUENCY == 0:
        print(f"Episode {episode + 1}: Finished training, Steps {step}")

# Evaluate the agent's performance
# test_episodes = 10
# episode_rewards = []

# for episode in range(test_episodes):
    # state = env.reset()[0]
    # episode_reward = 0
    # done = False

    # while not done:
        # action = new_agent.choose_action(state, eps=0.)
        # next_state, reward, done, truncated, _ = env.step(action)
        # episode_reward += reward
        # state = next_state
    # episode_rewards.append(episode_reward)

# average_reward = sum(episode_rewards) / test_episodes
# print(f"Average reward over {test_episodes} test episodes: {average_reward:.2f}")

# Visualize the agent's performance

env = gym.make(GAME_ID, render_mode=RENDER_GAME)

for rend in range(2):
    state = env.reset()[0]
    reward_total = 0
    done = False
    while not done:
        action = new_agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward_total += reward
        state = next_state
        # time.sleep(0.1)  # Add a delay to make the visualization easier to follow
    print(f"render test {rend} reward {reward_total:.2f}")

env.close()
