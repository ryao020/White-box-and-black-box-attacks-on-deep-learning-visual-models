import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Define environment
env = gym.make('CartPole-v1')

# Define constants
INPUT_SIZE = env.observation_space.shape[0]  # Assuming observation is a single array
OUTPUT_SIZE = env.action_space.n

# Define neural network
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)

# Train agent function
def train_agent():
    # Define network
    policy_net = PolicyNetwork(INPUT_SIZE, OUTPUT_SIZE)
    # Define optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=0.01)

    for episode in range(1000):
        # Reset environment
        state = env.reset()
        total_reward = 0

        while True:
            # Extract observation (handle tuple case)
            if isinstance(state, tuple):
                observation = state[0]
            else:
                observation = state

            # 处理状态
            processed_state = torch.from_numpy(observation).float().view(1, -1)

            # 在环境中采取动作
            action_probs = policy_net(processed_state)
            action = torch.multinomial(action_probs, 1).item()

            # 执行动作并获得奖励
            step_result = env.step(action)

            # 处理返回结果
            if isinstance(step_result, tuple) and len(step_result) >= 3:
                next_state, reward, done = step_result[:3]
            else:
                raise ValueError("Invalid environment step result")

            # 处理下一个状态
            processed_next_state = torch.from_numpy(next_state).float().view(1, -1)

            # 计算损失并进行反向传播
            optimizer.zero_grad()
            action_log_probs = torch.log(action_probs[0, action])
            loss = -action_log_probs * reward
            loss.backward()
            optimizer.step()

            total_reward += reward

            if done:
                break

            state = next_state

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # 保存训练好的模型
    torch.save(policy_net.state_dict(), 'policy_net_1.pth')

if __name__ == "__main__":
    train_agent()
