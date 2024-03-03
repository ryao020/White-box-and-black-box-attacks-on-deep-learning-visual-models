import gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

env = gym.make('CartPole-v1')

# 定义神经网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

# 训练代理的函数
def train_agent():
    input_size = env.observation_space.shape[0]
    output_size = env.action_space.n

    policy_net = PolicyNetwork(input_size, output_size)
    optimizer = optim.Adam(policy_net.parameters(), lr=0.001)

    for episode in range(1000):
        state = env.reset()
        total_reward = 0

        # 预处理状态
        processed_state = torch.tensor(state[0], dtype=torch.float32).view(1, -1)

        while True:
            # 在环境中采取动作
            action_probs = policy_net(processed_state)

            # 执行动作并获得奖励
            action = torch.multinomial(action_probs, 1).item()

            # 执行动作并获得奖励
            next_state, reward, done, _, _ = env.step(action)

            # 预处理下一个状态
            processed_next_state = torch.tensor(next_state, dtype=torch.float32).view(1, -1)

            # 计算损失并进行反向传播
            optimizer.zero_grad()
            action_log_probs = torch.log(action_probs[0, action])
            loss = -action_log_probs * reward
            loss.backward()
            optimizer.step()

            total_reward += reward
            processed_state = processed_next_state

            if done:
                break

        print(f"Episode: {episode + 1}, Total Reward: {total_reward}")

    # 保存训练好的模型
    torch.save(policy_net.state_dict(), 'policy_net.pth')

if __name__ == "__main__":
    train_agent()
