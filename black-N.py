import gym
import numpy as np
import torch
import torch.nn as nn
import os
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
ROOT_FOLDER = "root"  # 指定保存图表的根文件夹

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

# 新定义模型，与之前保存的模型结构一致
pytorch_model = PolicyNetwork(input_size, output_size)

# 加载重新训练的权重模型
checkpoint = torch.load("policy_net.pth")

# 从checkpoint中提取模型参数
model_state_dict = checkpoint

# 手动调整权重大小以匹配当前模型的结构
model_state_dict['fc.weight'] = model_state_dict['fc.weight'][:output_size, :input_size]
model_state_dict['fc.bias'] = model_state_dict['fc.bias'][:output_size]

# 加载模型权重
pytorch_model.load_state_dict(model_state_dict, strict=False)

# 创建用于生成对抗性示例的模型
adversarial_model = PolicyNetwork(input_size, output_size)

# 加载重新训练的权重模型
checkpoint = torch.load("policy_net.pth")

# 从checkpoint中提取模型参数
model_state_dict = checkpoint

# 手动调整权重大小以匹配当前模型的结构
adversarial_model.fc.weight.data = model_state_dict['fc.weight'][:output_size, :input_size]
adversarial_model.fc.bias.data = model_state_dict['fc.bias'][:output_size]

# 创建用于测试转移攻击的模型
transfer_model = PolicyNetwork(input_size, output_size)
transfer_model.fc.weight.data = model_state_dict['fc.weight'][:output_size, :input_size]
transfer_model.fc.bias.data = model_state_dict['fc.bias'][:output_size]

# 在这里添加 normalize_state 函数的定义
def normalize_state(state, state_min, state_max):
    normalized_state = [(s - s_min) / (s_max - s_min) for s, s_min, s_max in zip(state, state_min, state_max)]
    return normalized_state

# 选择动作的函数
def choose_action(state, model, input_size):
    state = torch.tensor(state, dtype=torch.float32).view(1, -1)
    probs = model(state).detach().numpy()[0]
    action = np.random.choice(len(probs), p=probs)
    return action

def generate_adversarial_example(state, adv_model):
    try:
        # 检查状态是否为 None
        if state is None or torch.tensor(state).numel() == 0:
            print("Warning: State is None or has length 0. Skipping iteration.")
            return None

        # 转换状态为 PyTorch 张量，并保留梯度
        state = torch.tensor(state, dtype=torch.float32).view(1, -1).requires_grad_()

        # 检查是否成功转换
        if state is None:
            print("Warning: Failed to convert state to PyTorch tensor.")
            return None

        # 打印状态的内容
        print("State content:", state)

        # 获取模型输出
        output = adv_model(state)

        # 检查模型输出是否为 Tensor
        if not isinstance(output, torch.Tensor):
            print("Warning: Model output is not a Tensor. Skipping iteration.")
            return None

        print("Model output:", output)  # 添加这行用于调试

        # 定义损失函数为交叉熵损失
        criterion = nn.CrossEntropyLoss()

        # 选择对抗性攻击方法
        if ATTACK_METHOD == "NI-FGSM":
            epsilon = 0.1
            alpha = 0.01

            # 获取模型的预测
            output = adv_model(state)

            print("Initial model output:", output)  # 添加这行用于调试

            target_label = torch.argmax(output)

            # 计算损失并反向传播
            loss = criterion(output, target_label.unsqueeze(0))
            loss.backward(retain_graph=True)  # 保留计算图

            # 检查梯度是否成功计算
            if state.grad is None:
                print("Warning: Gradient is None. Skipping iteration.")
                return None

            # 生成对抗性样本
            state = state + alpha * state.grad.sign()
            state = torch.clamp(state, 0, 1)

            # 清空梯度
            if state.grad is not None:
                state.grad.detach_()
                state.grad.zero_()

            # 再次传递对抗性样本给模型进行推断
            output = adv_model(state)

            print("Final adversarial output:", output)  # 添加这行用于调试

            # 返回对抗性样本
            return state.detach().numpy()[0]
        elif ATTACK_METHOD == "PGD":
            epsilon = 0.01
            alpha = 0.001

            for i in range(ITERATIONS):
                # 获取模型的预测
                output = adv_model(state)

                # 计算损失并反向传播
                loss = -criterion(output, target_label)  # 注意这里使用负的交叉熵损失
                loss.backward()

                # 生成对抗性样本
                state = state - alpha * state.grad.sign()
                state = torch.clamp(state, 0, 1)

                # 清空梯度
                adv_model.zero_grad()

            # 返回对抗性样本
            return state.detach().numpy()[0]
        else:
            #print(f"Invalid attack method: {ATTACK_METHOD}")
            return state.detach().numpy()[0]

    except Exception as e:
        print(f"Error during model inference: {e}")
        return None






def transfer_attack(state, model, adversarial_model):
    # 创建模型的深度复制，用于测试转移攻击
    test_model = PolicyNetwork(input_size, output_size)
    test_model.load_state_dict(model.state_dict())
    test_model.eval()

    # 转换状态为 PyTorch 张量，并保留梯度
    state = torch.tensor(state, dtype=torch.float32, requires_grad=True).view(1, -1)

    # 创建新的状态张量，允许梯度
    state_with_grad = torch.tensor(state, dtype=torch.float32).detach().requires_grad_(True).view(1, -1)

    # 选择转移攻击方法
    if ATTACK_METHOD == "PGD":
        epsilon = 0.01
        alpha = 0.001

        for i in range(ITERATIONS):
            # 获取模型的预测
            output = test_model(state_with_grad)

            # 计算损失并反向传播
            loss = -criterion(output, target_label)  # 注意这里使用负的交叉熵损失
            loss.backward()

            # 生成对抗性样本
            state_with_grad = state_with_grad - alpha * state_with_grad.grad.sign()
            state_with_grad = torch.clamp(state_with_grad, 0, 1)

            # 清空梯度
            test_model.zero_grad()

        # 返回对抗性样本
        return state_with_grad.detach().numpy()[0]
    else:
        #print(f"Invalid attack method: {ATTACK_METHOD}")
        return state.detach().numpy()[0]


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

            if len(next_state) == 0:
                print("Warning: Next state has length 0.")
            else:
                # 修复此处代码
                adv_state = generate_adversarial_example(next_state, adversarial_model)
                if adv_state is None:
                    continue  # 跳过本次循环

            # 测试转移攻击
            transfer_result = transfer_attack(next_state, model, adversarial_model)

            total_reward += reward
            steps += 1

            if done:
                print(f"Episode {i}, Total Reward: {total_reward}")
                break

            state = next_state


# 修改后的可视化和分析函数
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

            if len(state) == 0:
                print("Warning: Initial state has length 0.")
            else:
                # 调用 generate_adversarial_example 函数
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

    # 保存图表
    save_folder = os.path.join(ROOT_FOLDER, "visualization")
    os.makedirs(save_folder, exist_ok=True)
    save_path = os.path.join(save_folder, "reward_comparison.png")

    # 可视化结果
    plt.plot(original_rewards, label='Original Model')
    plt.plot(adversarial_rewards, label='Adversarial Model')
    plt.plot(transfer_rewards, label='Transfer Model')
    plt.xlabel('Episodes')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig(save_path)  # 保存图表
    plt.show()

if __name__ == "__main__":
# 运行黑盒攻击
    input_size = env.observation_space.shape[0]
    black_box_attack(transfer_model, env, EPISODES, ATTACK_METHOD, input_size)

    # 可视化结果
    visualize_results(pytorch_model, adversarial_model, transfer_model, env, EPISODES, input_size)
