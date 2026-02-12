"""dql_cartpole"""
# Import necessary libraries
# https://github.com/SeeknnDestroy/DQN-CartPole/blob/master/dql-cartpole.ipynb
import random
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
import gymnasium as gym

SEED=170715
NUM_TRAINING_EPISODES = 250
# NUM_TRAINING_EPISODES = 20
MAX_STEPS_PER_EPISODE = 200
EPSILON_START = 1.0
EPSILON_END = 0.2
EPSILON_DECAY_RATE = 0.99
GAMMA = 0.9
LEARNING_RATE = 0.0025
REPLAY_BUFFER_SIZE = 10000
BATCH_SIZE = 128
UPDATE_FREQUENCY = 10
FC1_UNITS = 64
FC2_UNITS = 64
TAU = 0.001
GAME_ID = "CartPole-v1"
SHOW_GAME = "human"
NUM_RENDER_TESTS = 4

# Check if GPU is available
device = torch.device( "cuda" if torch.cuda.is_available() else "cpu")
print(f"Device detected: {device}")

class DNQ(nn.Module):
    """Deep Q network"""
    def __init__(self, state_size, action_size):
        super().__init__()
        self.seed = torch.manual_seed(SEED)
        self.fc1 = nn.Linear(state_size, FC1_UNITS)
        self.fc2 = nn.Linear(FC1_UNITS, FC2_UNITS)
        self.fc3 = nn.Linear(FC2_UNITS, action_size)
        self.to(device)

    def forward(self, state_l):
        """ forward pass"""
        x = F.relu(self.fc1(state_l))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """Deep Q network Agent"""
    def __init__(self, env_l):
        self.input_dim = env_l.observation_space.shape[0]
        self.output_dim = env_l.action_space.n
        random.seed(SEED)
        self.policy_net = DNQ(self.input_dim, self.output_dim).to(device)
        self.target_net = DNQ(self.input_dim, self.output_dim).to(device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)

    def choose_action(self, state_l, eps=0):
        """Choose an action based on the current state"""
        state_tensor = torch.from_numpy(state_l).float().unsqueeze(0).to(device)

        self.policy_net.eval()
        with torch.no_grad():
            action_values = self.policy_net(state_tensor)
        self.policy_net.train()

        if np.random.random() > eps:
            return action_values.argmax(dim=1).item()
        return np.random.randint(self.output_dim)

    def optimize_model(self, transition_samples_l):
        """Learn from a sample batch of experiences"""
        states, actions, rewards, next_states, dones = zip(*transition_samples_l)
        # states = torch.from_numpy(np.vstack(states)).float().to(device)
        # actions = torch.from_numpy(np.vstack(actions)).long().to(device)
        # rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        # next_states = torch.from_numpy(np.vstack(next_states)).float().to(device)
        # dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)
        states = torch.from_numpy(np.asarray(states)).to(device)
        actions = torch.from_numpy(np.vstack(actions)).to(device)
        rewards = torch.from_numpy(np.vstack(rewards)).float().to(device)
        next_states = torch.from_numpy(np.asarray(next_states)).to(device)
        dones = torch.from_numpy(np.vstack(dones).astype(np.uint8)).float().to(device)

        q_targets_next = self.target_net(next_states).detach().max(1)[0].unsqueeze(1)
        q_targets = rewards + (GAMMA * q_targets_next * (1 - dones))

        q_expected = self.policy_net(states).gather(1, actions)

        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.soft_update(self.policy_net, self.target_net)

    def soft_update(self, local_model, target_model):
        """soft update"""
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(TAU * local_param.data + (1.0 - TAU) * target_param.data)

# Set up the environment
env = gym.make("CartPole-v1", MAX_STEPS_PER_EPISODE, render_mode=SHOW_GAME)

# Initialize the DQNAgent
agent = DQNAgent(env)

# store all action, obvservation data
replay_memory = []

# Training loop
for episode in range(NUM_TRAINING_EPISODES):
    # Reset the environment
    state = env.reset()[0]
    epsilon = max(EPSILON_END, EPSILON_START * (EPSILON_DECAY_RATE ** episode))

    step = 0
    done = False
    while not done:
        # Choose and perform an action
        action = agent.choose_action(state, epsilon)
        next_state, reward, done, truncated, _ = env.step(action)

        replay_memory.append((state, action, reward, next_state, done))

        if len(replay_memory) >= BATCH_SIZE:
            transition_samples = random.sample(replay_memory, BATCH_SIZE)
            # Update the agent's knowledge
            agent.optimize_model(transition_samples)

        state = next_state
        step += 1

    if (episode + 1) % UPDATE_FREQUENCY == 0:
        print(f"Episode {episode + 1}: Finished training, Steps {step}")

# Render test
input('Please a key to start the render test')
env = gym.make(GAME_ID, render_mode=SHOW_GAME)

for rend in range(NUM_RENDER_TESTS):
    state = env.reset()[0]
    reward_total = 0
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, truncated, _ = env.step(action)
        reward_total += reward
        state = next_state
        # time.sleep(0.1)  # Add a delay to make the visualization easier to follow
    print(f"render test {rend} reward {reward_total:.2f}")

env.close()
