import gym
import torch
import torch.nn as nn
import numpy as np
from gym import __version__ as gym_version
print(f"Gym version: {gym_version}")

# 定义环境
env = gym.make('CartPole-v1')

# 定义常量
INPUT_SIZE = env.observation_space.shape[0]
OUTPUT_SIZE = env.action_space.n

# 定义神经网络
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

# 加载预训练模型
pretrained_model = PolicyNetwork(INPUT_SIZE, OUTPUT_SIZE)
pretrained_model.load_state_dict(torch.load('policy_net.pth'))

# 修改评估函数
def evaluate_model(model, episodes=10):
    total_rewards = []
    for episode in range(episodes):
        state = env.reset()
        total_reward = 0
        while True:
            if isinstance(state, tuple):
                processed_state = torch.from_numpy(state[0]).float().view(1, -1)
            else:
                processed_state = torch.from_numpy(state).float().view(1, -1)

            action_probs = model(processed_state)
            action = torch.argmax(action_probs).item()

            # 更新以处理返回值的变化
            step_result = env.step(action)
            state, reward, done, _ = step_result[0], step_result[1], step_result[2], step_result[3]



            total_reward += reward
            if done:
                break
        total_rewards.append(total_reward)
    return total_rewards

# 评估预训练模型
pretrained_rewards = evaluate_model(pretrained_model, episodes=10)
print(f"预训练模型平均奖励: {np.mean(pretrained_rewards)}")

# 加载新训练的模型
newly_trained_model = PolicyNetwork(INPUT_SIZE, OUTPUT_SIZE)
newly_trained_model.load_state_dict(torch.load('policy_net.pth'))

# 评估新训练的模型
newly_trained_rewards = evaluate_model(newly_trained_model, episodes=10)
print(f"新训练模型平均奖励: {np.mean(newly_trained_rewards)}")

# 检查权重是否相同
weights_match = torch.all(torch.eq(pretrained_model.fc.weight, newly_trained_model.fc.weight))
print(f"权重匹配: {weights_match}")
