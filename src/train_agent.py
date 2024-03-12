import gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Hyperparameters
learning_rate = 0.001
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.01
epsilon_decay = 200
memory_size = 10000
batch_size = 64
target_update = 10

# Environment setup
env = gym.make('CartPole-v1')
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# DQN model
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 256),  # Increased from 128 to 256
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def forward(self, x):
        return self.fc(x)

# Initialize networks
policy_net = DQN(state_dim, action_dim).to(device)
target_net = DQN(state_dim, action_dim).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

# Optimizer
optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)

# Experience replay memory
memory = deque(maxlen=memory_size)


# Epsilon-greedy action selection
def select_action(state, epsilon):
    # Ensure state is a numpy array with consistent shape
    if not isinstance(state, np.ndarray):
        state = np.asarray(state)

    # Check if state is already 2D; if not, reshape it
    if state.ndim == 1:
        state = state.reshape(1, -1)

    # Convert to PyTorch tensor and move to the specified device
    state_tensor = torch.FloatTensor(state).to(device)

    if random.random() > epsilon:
        with torch.no_grad():
            # Forward pass through the network
            action_values = policy_net(state_tensor)
            # Select the action with the highest value
            return action_values.max(1)[1].view(1, 1).item()
    else:
        # Select a random action
        return random.randrange(action_dim)


# Save experience
def store_transition(state, action, reward, next_state, done):
    memory.append((state, action, reward, next_state, done))


# Sample mini-batch
def sample_batch():
    transitions = random.sample(memory, batch_size)
    batch = list(zip(*transitions))
    states, actions, rewards, next_states, dones = batch
    return (torch.FloatTensor(states).to(device), torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device), torch.FloatTensor(next_states).to(device),
            torch.BoolTensor(dones).to(device))

# DQN learning algorithm
def optimize_model():
    if len(memory) < batch_size:
        return
    states, actions, rewards, next_states, dones = sample_batch()

    q_values = policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0].detach()
    expected_q_values = rewards + gamma * next_q_values * (~dones)

    loss = nn.MSELoss()(q_values, expected_q_values)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


# Set random seeds for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env.seed(seed)

def main():
    # set_seed(0)
    set_seed(42)

    # Training loop
    num_episodes = 1000
    epsilon = epsilon_start

    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        while True:
            epsilon = max(epsilon_end, epsilon_start - episode / epsilon_decay)
            action = select_action(state, epsilon)
            next_state, reward, done, _ = env.step(action)
            store_transition(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

            optimize_model()

            if done:
                print(f"Episode: {episode}, Total Reward: {total_reward}")
                break

        total_rewards.append(total_reward)

        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # save the model
    # torch.save(policy_net.state_dict(), 'policy_net_dqn1.pth')
    torch.save(policy_net.state_dict(), 'policy_net_dqn2.pth')

    plt.plot(total_rewards)
    # plt.savefig('training_progress_dqn1.png')
    plt.savefig('training_progress_dqn2.png')
    # visualize the training progress

    env.close()

if __name__ == "__main__":
    main()