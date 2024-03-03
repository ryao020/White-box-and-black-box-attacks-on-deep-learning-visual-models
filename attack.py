# 导入 gym 库，用于提供游戏环境
import gym
# 导入 torch 库，用于构建和训练深度学习模型
import torch
# 导入 numpy 库，用于进行数值计算
import numpy as np
# 导入 torchvision 模块，用于提供一些图像处理的工具
from torchvision import transforms
# 定义游戏环境的名称，您可以根据您的选择进行修改
ENV_NAME = "CartPole-v1"
# 定义模型的超参数，您可以根据您的需要进行调整
LEARNING_RATE = 0.001 # 学习率
GAMMA = 0.99 # 折扣因子
BATCH_SIZE = 32 # 批大小
EPISODES = 1000 # 训练的迭代次数
# 定义攻击的方法和参数，您可以根据您的选择进行修改
ATTACK_METHOD = "FGSM" # 攻击的方法，可选 FGSM, MI-FGSM, NI-FGSM 等
EPSILON = 0.01 # 攻击的扰动大小，即每个像素的最大变化量
ALPHA = 0.001 # 攻击的步长，即每次更新的变化量
ITERATIONS = 10 # 攻击的迭代次数，即更新对抗样本的次数
# 创建一个游戏环境的对象
env = gym.make(ENV_NAME)
# 获取游戏的动作空间的大小，即可以执行的动作的数量
n_actions = env.action_space.n
# 获取游戏的状态空间的形状，即每个状态的特征的数量
state_shape = env.observation_space.shape
# 创建一个深度学习模型的对象，使用 torch.nn.Module 类
class PolicyNetwork(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(PolicyNetwork, self).__init__()
        self.fc = torch.nn.Linear(input_size, output_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc(x)
        return self.softmax(x)

# 实例化一个模型的对象，使用之前定义的类
model = PolicyNetwork(state_shape[0], n_actions)
# 定义一个优化器，使用 torch.optim.Adam 类
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
# 加载您之前训练好的模型的权重，您可以根据您的需要选择不同的权重文件
model.load_state_dict(torch.load('policy_net.pth'))
# 定义一个函数，用于根据模型的输出选择一个动作
def choose_action(state):
  # 将状态转换为模型的输入格式，即一个二维的张量
  state = torch.tensor(state, dtype=torch.float32).view(1, -1)
  # 使用模型预测状态对应的动作的概率
  probs = model(state)
  # 使用轮盘赌的方法，根据概率选择一个动作
  action = torch.multinomial(probs, 1).item()
  # 返回选择的动作
  return action
# 定义一个函数，用于根据攻击的方法和参数生成对抗样本
def generate_adversarial_example(state):
  # 将状态转换为模型的输入格式，即一个二维的张量
  state = torch.tensor(state, dtype=torch.float32).view(1, -1)
  # 创建一个变量，用于存储对抗样本，初始值为原始状态
  adv_state = torch.autograd.Variable(state, requires_grad=True)
  # 根据不同的攻击的方法，执行不同的操作
  if ATTACK_METHOD == "FGSM":
    # 使用 FGSM 方法，即快速梯度符号法，只进行一次更新
    # 计算状态对应的损失值，即交叉熵损失
    loss = torch.nn.functional.cross_entropy(model(state), model(adv_state))
    # 计算损失值对对抗样本的梯度，即损失值的变化率
    grad = torch.autograd.grad(loss, adv_state)[0]
    # 根据梯度的符号和扰动大小，更新对抗样本
    adv_state = adv_state + EPSILON * torch.sign(grad)
    # 将对抗样本的值限制在合法的范围内，即 0 到 1
    adv_state = torch.clamp(adv_state, 0, 1)
  elif ATTACK_METHOD == "MI-FGSM":
    # 使用 MI-FGSM 方法，即动量迭代快速梯度符号法，进行多次更新
    # 创建一个变量，用于存储动量，初始值为 0
    momentum = torch.zeros_like(adv_state)
    # 循环进行指定的迭代次数
    for i in range(ITERATIONS):
      # 计算状态对应的损失值，即交叉熵损失
      loss = torch.nn.functional.cross_entropy(model(state), model(adv_state))
      # 计算损失值对对抗样本的梯度，即损失值的变化率
      grad = torch.autograd.grad(loss, adv_state)[0]
      # 根据梯度和动量的加权平均，更新动量
      momentum = GAMMA * momentum + grad / torch.mean(torch.abs(grad))
      # 根据动量的符号和扰动大小，更新对抗样本
      adv_state = adv_state + EPSILON * torch.sign(momentum)
      # 将对抗样本的值限制在合法的范围内，即 0 到 1
      adv_state = torch.clamp(adv_state, 0, 1)
  elif ATTACK_METHOD == "NI-FGSM":
      # 使用 NI-FGSM 方法，即噪声注入快速梯度符号法，进行多次更新
      # 创建一个变量，用于存储噪声，初始值为 0
      noise = torch.zeros_like(adv_state)
      # 循环进行指定的迭代次数
      for i in range(ITERATIONS):
        # 计算状态对应的损失值，即交叉熵损失
        loss = torch.nn.functional.cross_entropy(model(state), model(adv_state))
        # 计算损失值对对抗样本的梯度，即损失值的变化率
        grad = torch.autograd.grad(loss, adv_state)[0]
        # 根据梯度和噪声的加权平均，更新噪声
        noise = GAMMA * noise + grad / torch.mean(torch.abs(grad))
        # 根据噪声的符号和扰动大小，更新对抗样本
        adv_state = adv_state + EPSILON * torch.sign(noise)
        # 将对抗样本的值限制在合法的范围内
# 定义一个函数，用于将值限制在 [0, 1] 范围内
def clip(x):
    return torch.clamp(x, 0, 1)

# 定义一个函数，用于生成对抗样本
def generate_adversarial_example(state):
    # 将状态转换为张量，并设置requires_grad为True，表示需要计算梯度
    state = torch.tensor(state, dtype=torch.float32, requires_grad=True)
    # 根据状态选择一个动作
    action = choose_action(state)
    # 计算动作的对数概率，即模型的输出
    log_prob = model(state)[action]
    # 计算动作的负对数概率，即损失函数
    loss = -log_prob
    # 计算损失函数对状态的梯度
    loss.backward()
    # 获取状态的梯度
    grad = state.grad.data
    # 根据梯度的符号和扰动值epsilon，生成对抗样本
    adv_state = state + EPSILON * grad.sign()
    # 将对抗样本的值限制在合法的范围内
    adv_state = clip(adv_state)
    # 返回对抗样本
    return adv_state

# 定义一个函数，用于测试模型在正常状态和对抗状态下的表现，即平均奖励和成功率
def test_model(adv=False):
    # 创建一个变量，用于存储累积奖励
    total_reward = 0
    # 创建一个变量，用于存储成功次数
    success_count = 0
    # 循环进行指定的测试次数
    for i in range(100):
        # 重置游戏环境，获取初始状态
        state, _ = env.reset()
        # 创建一个变量，用于存储单次奖励
        reward = 0
        # 循环进行游戏，直到结束
        while True:
            # 如果使用对抗样本，调用生成对抗样本的函数，替换原始状态
            if adv:
                state = generate_adversarial_example(state)
            # 根据状态选择一个动作
            action = choose_action(state)
            # 在游戏环境中执行动作，获取下一个状态，奖励，是否结束，和其他信息
            next_state, reward, done, _, _ = env.step(action)
            # 累加奖励
            total_reward += reward
            # 更新状态
            state = next_state
            # 如果游戏结束，跳出循环
            if done:
                break
        # 如果单次奖励大于等于 195，表示成功
        if total_reward >= 195:
            # 累加成功次数
            success_count += 1
    # 计算平均奖励，即累积奖励除以测试次数
    avg_reward = total_reward / 100
    # 计算成功率，即成功次数除以测试次数
    success_rate = success_count / 100
    # 返回平均奖励和成功率
    return avg_reward, success_rate

# 创建一个列表，用于存储不同的扰动值 epsilon
epsilons = [0.01, 0.02, 0.03, 0.04, 0.05]
# 循环对每个 epsilon 进行实验
for epsilon in epsilons:
  # 打印当前的 epsilon 值
  print(f"Epsilon: {epsilon}")
  # 更新全局变量 EPSILON 的值，用于生成对抗样本
  EPSILON = epsilon
  # 调用测试模型的函数，获取正常状态下的平均奖励和成功率
  normal_reward, normal_rate = test_model(adv=False)
  # 打印正常状态下的平均奖励和成功率
  print(f"Normal Reward: {normal_reward}, Normal Rate: {normal_rate}")
  # 调用测试模型的函数，获取对抗状态下的平均奖励和成功率
  adv_reward, adv_rate = test_model(adv=True)
  # 打印对抗状态下的平均奖励和成功率
  print(f"Adversarial Reward: {adv_reward}, Adversarial Rate: {adv_rate}")
  # 打印空行，用于分隔不同的实验
  print()
