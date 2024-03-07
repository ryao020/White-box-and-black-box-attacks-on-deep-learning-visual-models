import gym
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# 全局配置
ENV_NAME = "CartPole-v1"
LEARNING_RATE = 0.01
GAMMA = 0.99
BATCH_SIZE = 32
EPISODES = 100
ATTACK_METHOD = "NI-FGSM"
EPSILON = 0.01
ALPHA = 0.001
ITERATIONS = 10

# 创建环境
env = gym.make(ENV_NAME)
n_actions = env.action_space.n
state_shape = env.observation_space.shape
input_size = state_shape[0]
output_size = env.action_space.n

# 定义策略网络
class PolicyNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = nn.Linear(input_size, output_size)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

# 加载原始模型
pytorch_model = PolicyNetwork(input_size, output_size)
pytorch_model.load_state_dict(torch.load("policy_net.pth"))

# 创建用于生成对抗性示例的模型
adversarial_model = PolicyNetwork(input_size, output_size)
adversarial_model.load_state_dict(torch.load("policy_net.pth"))

# 创建用于测试转移攻击的模型
transfer_model = PolicyNetwork(input_size, output_size)
transfer_model.load_state_dict(torch.load("policy_net.pth"))

# 选择动作的函数
def choose_action(state, model, input_size):
    state = torch.tensor(state, dtype=torch.float32).view(1, -1)
    probs = model(state).detach().numpy()[0]
    action = np.random.choice(len(probs), p=probs)
    return action

def generate_adversarial_example(state, model):
    # 创建模型的深度复制，用于生成对抗性示例
    adv_model = PolicyNetwork(input_size, output_size)
    adv_model.load_state_dict(model.state_dict())
    adv_model.eval()
    # 确保状态不为 None
    if state is None:
        # 处理状态为 None 的情况，可以选择跳过当前迭代或采取其他措施
        return None

    # 转换状态为 PyTorch 张量，并保留梯度
    state = torch.tensor(state, dtype=torch.float32).view(1, -1).requires_grad_().retain_grad()

    # 获取模型输出
    output = adv_model(state)

    # 确保输出不为 None
    if output is None:
        # 处理输出为 None 的情况，可以选择跳过当前迭代或采取其他措施
        return None

    # 检查模型输出是否为 Tensor
    if not isinstance(output, torch.Tensor):
        # 处理输出不为 Tensor 的情况，可以选择跳过当前迭代或采取其他措施
        return None

    # 定义损失函数为交叉熵损失
    criterion = nn.CrossEntropyLoss()

    # 选择对抗性攻击方法
    if ATTACK_METHOD == "NI-FGSM":
        epsilon = 0.01
        alpha = 0.001

        for i in range(ITERATIONS):
            # 获取模型的预测
            output = adv_model(state)
            target_label = torch.argmax(output)

            # 计算损失并反向传播
            loss = criterion(output, target_label.unsqueeze(0))
            loss.backward(retain_graph=True)  # 保留计算图

            # 检查梯度是否成功计算
            if state.grad is None:
                print("Warning: Gradient is None. Skipping iteration.")
                break

            # 生成对抗性样本
            state = state + alpha * state.grad.sign()
            state = torch.clamp(state, 0, 1)

            # 清空梯度
            state.grad.zero_()

        # 返回对抗性样本
        return state.detach().numpy()[0]
    else:
        print(f"Invalid attack method: {ATTACK_METHOD}")
        return state.detach().numpy()[0]



# 测试转移攻击的函数
def transfer_attack(state, model, adversarial_model):
    # 创建模型的深度复制，用于测试转移攻击
    test_model = PolicyNetwork(input_size, output_size)
    test_model.load_state_dict(model.state_dict())
    test_model.eval()

    # 转换状态为 PyTorch 张量
    state = torch.tensor(state, dtype=torch.float32, requires_grad=True).view(1, -1)

    state.requires_grad = True

    # 选择转移攻击方法
    if ATTACK_METHOD == "PGD":
        epsilon = 0.01
        alpha = 0.001

        for i in range(ITERATIONS):
            # 获取模型的预测
            output = test_model(state)

            # 计算损失并反向传播
            loss = -criterion(output, target_label)  # 注意这里使用负的交叉熵损失
            loss.backward()

            # 生成对抗性样本
            state = state - alpha * state.grad.sign()
            state = torch.clamp(state, 0, 1)

            # 清空梯度
            test_model.zero_grad()

        # 返回对抗性样本
        return state.detach().numpy()[0]
    else:
        print(f"Invalid attack method: {ATTACK_METHOD}")
        return state.detach().numpy()[0]

# 黑盒攻击的函数
def black_box_attack(model, env, episodes, attack_method, input_size):
    for i in range(episodes):
        state = env.reset()
        state = state[0]
        total_reward = 0
        steps = 0

        while True:
            action = choose_action(state, model, input_size)
            step_result = env.step(action)
            next_state, reward, done, _ = step_result[:4]

            # 生成对抗性示例
            adv_state = generate_adversarial_example(state, adversarial_model)

            # 测试转移攻击
            transfer_result = transfer_attack(state, model, adversarial_model)

            total_reward += reward
            steps += 1

            if done:
                print(f"Episode {i}, Total Reward: {total_reward}")
                break

            state = next_state

# 添加可视化和分析代码
def visualize_results(original_model, adversarial_model, transfer_model, env, episodes, input_size):
    original_rewards = []
    adversarial_rewards = []
    transfer_rewards = []

    for i in range(episodes):
        state = env.reset()
        state = state[0]
        total_reward_original = 0
        total_reward_adversarial = 0
        total_reward_transfer = 0

        while True:
            # 原始模型
            action_original = choose_action(state, original_model, input_size)
            step_result_original = env.step(action_original)
            _, reward_original, done_original, _ = step_result_original[:4]
            total_reward_original += reward_original

            # 对抗性模型
            adv_state = generate_adversarial_example(state, adversarial_model)
            action_adversarial = choose_action(adv_state, original_model, input_size)
            step_result_adversarial = env.step(action_adversarial)
            _, reward_adversarial, done_adversarial, _ = step_result_adversarial[:4]
            total_reward_adversarial += reward_adversarial

            # 转移攻击模型
            transfer_result = transfer_attack(state, original_model, adversarial_model)
            action_transfer = choose_action(transfer_result, transfer_model, input_size)
            step_result_transfer = env.step(action_transfer)
            _, reward_transfer, done_transfer, _ = step_result_transfer[:4]
            total_reward_transfer += reward_transfer

            if done_original or done_adversarial or done_transfer:
                break

            state = step_result_original[0]

        original_rewards.append(total_reward_original)
        adversarial_rewards.append(total_reward_adversarial)
        transfer_rewards.append(total_reward_transfer)

    # 可视化结果
    plt.plot(original_rewards, label='Original Model')
    plt.plot(adversarial_rewards, label='Adversarial Model')
    plt.plot(transfer_rewards, label='Transfer Model')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    # 运行黑盒攻击
    input_size = env.observation_space.shape[0]
    black_box_attack(transfer_model, env, EPISODES, ATTACK_METHOD, input_size)

    # 可视化结果
    visualize_results(pytorch_model, adversarial_model, transfer_model, env, EPISODES, input_size)
