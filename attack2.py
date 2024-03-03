import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
import numpy as np
import random

# 设置随机种子，保证实验的可重复性
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

# 设置设备，如果有 GPU 则使用 GPU，否则使用 CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 设置超参数
EPSILONS = [0, .05, .1, .15, .2, .25, .3]  # 扰动值列表
PRETRAINED_MODEL = "checkpoint/lenet_mnist_model.pth"  # 预训练模型路径
BATCH_SIZE = 1  # 批大小
NUM_WORKERS = 4  # 数据加载器的工作进程数
LEARNING_RATE = 0.01  # 学习率
NUM_EPOCHS = 100  # 迭代轮数
TARGET = 7  # 定向攻击的目标标签
C = 0.1  # C&W 攻击的常数

# 加载数据集，使用 MNIST 测试集
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = torchvision.datasets.MNIST(root='./datasets', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

# 定义 LeNet 模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 初始化模型，并加载预训练的权重
model = LeNet().to(device)
model.load_state_dict(torch.load(PRETRAINED_MODEL, map_location='cpu'))

# 设置模型为评估模式，不进行梯度更新
model.eval()

# 定义 FGSM 攻击函数
def fgsm_attack(image, epsilon, data_grad):
    # 根据梯度的符号和扰动值 epsilon，生成对抗样本
    sign_data_grad = data_grad.sign()
    perturbed_image = image + epsilon * sign_data_grad
    # 将对抗样本的值限制在合法的范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本
    return perturbed_image

# 定义 MI-FGSM 攻击函数
def mi_fgsm_attack(image, epsilon, data_grad, decay_factor, g):
    # 根据梯度的符号、扰动值 epsilon 和衰减因子 decay_factor，生成对抗样本
    sign_data_grad = data_grad.sign()
    g = decay_factor * g + sign_data_grad
    perturbed_image = image + epsilon * g
    # 将对抗样本的值限制在合法的范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本和累积梯度
    return perturbed_image, g

# 定义 NI-FGSM 攻击函数
def ni_fgsm_attack(image, epsilon, data_grad, decay_factor, g):
    # 根据梯度的符号、扰动值 epsilon 和衰减因子 decay_factor，生成对抗样本
    sign_data_grad = data_grad.sign()
    g = decay_factor * g + sign_data_grad / torch.norm(sign_data_grad, p=1)
    perturbed_image = image + epsilon * g
    # 将对抗样本的值限制在合法的范围内
    perturbed_image = torch.clamp(perturbed_image, 0, 1)
    # 返回对抗样本和累积梯度
    return perturbed_image, g

# 定义 JSMA 攻击函数
def jsma_attack(image, target, model, theta, gamma):
    # 将图像转换为张量，并设置 requires_grad 为 True，表示需要计算梯度
    image = torch.tensor(image, dtype=torch.float32, requires_grad=True)
    # 初始化对抗样本为原始图像
    adv_jsma = image.clone()
    # 获取图像的形状
    n, c, h, w = image.shape
    # 获取图像的像素索引
    pixels = [(i, j) for i in range(h) for j in range(w)]
    # 对像素索引进行随机打乱
    random.shuffle(pixels)
    # 计算目标类别的概率
    target_prob = model(adv_jsma)[0, target]
    # 设置循环终止条件
    stop = False
    # 对每个像素进行遍历
    for i, j in pixels:
        # 如果目标类别的概率已经超过阈值 gamma，或者对抗样本与原始图像的差异已经超过阈值 theta，或者循环终止标志为真，则停止循环
        if target_prob > gamma or torch.norm(adv_jsma - image) > theta or stop:
            break
        # 对每个通道进行遍历
        for k in range(c):
            # 保存当前像素的值
            original_value = adv_jsma[0, k, i, j].item()
            # 将当前像素的值增加 theta
            adv_jsma[0, k, i, j] = original_value + theta
            # 计算增加后的像素对目标类别的梯度和其他类别的梯度
            adv_jsma.requires_grad = True
            out = model(adv_jsma)
            target_grad = torch.autograd.grad(out[0, target], adv_jsma, retain_graph=True)[0][0, k, i, j].item()
            other_grad = torch.autograd.grad(torch.sum(out[0, :target] + out[0, target + 1:]), adv_jsma)[0][0, k, i, j].item()
            # 恢复当前像素的值
            adv_jsma[0, k, i, j] = original_value
            # 计算增加后的像素对目标类别的梯度和其他类别的梯度的乘积
            saliency = target_grad * other_grad
            # 如果该乘积为正，说明增加该像素的值可以增加目标类别的概率并减少其他类别的概率，那么就选择该像素进行修改
            if saliency > 0:
                # 将当前像素的值增加 theta，并将其限制在 [0, 1] 的范围内
                adv_jsma[0, k, i, j] = torch.clamp(original_value + theta, 0, 1)
                # 重新计算目标类别的概率
                target_prob = model(adv_jsma)[0, target]
                # 如果目标类别的概率已经达到 1，说明攻击成功，设置循环终止标志为真
                if target_prob == 1:
                    stop = True
                # 跳出通道的循环
                break
    # 返回对抗样本
    return adv_jsma

# 简单黑盒攻击逻辑示例，随机选择像素进行修改
def simple_black_box_attack(image, epsilon):
    perturbed_image = image.clone()
    num_pixels = image.shape[-1]
    pixels_to_modify = np.random.choice(num_pixels, int(epsilon * num_pixels), replace=False)
    
    for pixel in pixels_to_modify:
        perturbed_image[0, 0, pixel // 28, pixel % 28] = torch.rand(1)
    
    return torch.clamp(perturbed_image, 0, 1)

# 测试黑盒攻击
for data, target in test_loader:
    data, target = data.to(device), target.to(device)
    data.requires_grad = True

    # 计算梯度
    output = model(data)
    loss = F.nll_loss(output, target)
    model.zero_grad()
    loss.backward()

    # FGSM 攻击
    perturbed_data = fgsm_attack(data, 0.1, data.grad.data)

    # MI-FGSM 攻击
    g = torch.zeros_like(data.grad.data)
    perturbed_data_mi, g = mi_fgsm_attack(data, 0.1, data.grad.data, 0.9, g)

    # NI-FGSM 攻击
    g = torch.zeros_like(data.grad.data)
    perturbed_data_ni, g = ni_fgsm_attack(data, 0.1, data.grad.data, 0.9, g)

    # JSMA 攻击
    perturbed_data_jsma = jsma_attack(data, TARGET, model, 0.1, 0.8)

    # 简单黑盒攻击
    perturbed_data_simple_black_box = simple_black_box_attack(data, 0.1)
