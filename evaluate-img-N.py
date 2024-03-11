import gym
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

ENV_NAME = "CartPole-v1"
LEARNING_RATE = 0.01
GAMMA = 0.99
BATCH_SIZE = 32
EPISODES = 1000
ATTACK_METHOD = "FGSM"
EPSILON = 0.01
ALPHA = 0.001
ITERATIONS = 10

env = gym.make(ENV_NAME)
n_actions = env.action_space.n
state_shape = env.observation_space.shape
input_size = state_shape[0]
output_size = env.action_space.n

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





def choose_action(state, model, input_size):
    state = torch.tensor(state, dtype=torch.float32).view(1, -1)
    probs = model(state).detach().numpy()[0]
    action = np.random.choice(len(probs), p=probs)
    return action

def generate_adversarial_example(state, model):
    model = PolicyNetwork(input_size, output_size)
    state = np.expand_dims(state, axis=0)
    adv_state = torch.tensor(state, dtype=torch.float32, requires_grad=True)

    if ATTACK_METHOD == "FGSM":
        logits = model(adv_state)
        target_label = np.random.randint(output_size)
        loss = torch.nn.functional.cross_entropy(logits, torch.tensor([target_label]))
        loss.backward()
        grad = adv_state.grad
        adv_state = adv_state + EPSILON * torch.sign(grad)
        adv_state = torch.clamp(adv_state, 0, 255)
    elif ATTACK_METHOD == "MI-FGSM":
        momentum = torch.zeros_like(adv_state)
        for i in range(ITERATIONS):
            loss = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
            loss.backward()
            grad = adv_state.grad
            momentum = GAMMA * momentum + grad / torch.mean(torch.abs(grad))
            adv_state = adv_state + EPSILON * torch.sign(momentum)
            adv_state = torch.clamp(adv_state, 0, 255)
    elif ATTACK_METHOD == "NI-FGSM":
        noise = torch.zeros_like(adv_state)
        for i in range(ITERATIONS):
            loss = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
            loss.backward()
            grad = adv_state.grad
            noise = GAMMA * noise + grad / torch.mean(torch.abs(grad))
            adv_state = adv_state + EPSILON * torch.sign(noise)
            adv_state = torch.clamp(adv_state, 0, 255)
    else:
        print(f"Invalid attack method: {ATTACK_METHOD}")
        return state

# Save adversarial example image
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original state")
    plt.plot(state[0])
    plt.subplot(1, 2, 2)
    plt.title("Adversarial state")
    plt.plot(adv_state.detach().numpy()[0])
    plt.savefig("/root/original_adversarial_states.png")

    return adv_state.detach().numpy()[0]

# 以下为新增的测量函数
def get_cpu_usage():
    # 替换为实际的 CPU 使用率测量代码
    return 0.0

def get_gpu_usage():
    # 替换为实际的 GPU 使用率测量代码
    return 0.0

def get_memory_usage():
    # 替换为实际的内存使用率测量代码
    return 0.0

# 修改 evaluate 函数以收集测量数据
baseline_metrics = {"CPU": [], "GPU": [], "Memory": []}
attack_metrics = {"CPU": [], "GPU": [], "Memory": []}

# Modify the evaluate function to fix adversarial example generation
def evaluate(model, env, episodes, attack_method, input_size):
    accuracies = []
    losses = []

    for i in range(episodes):
        state = env.reset()
        state = state[0]
        total_reward = 0
        steps = 0

        while True:
            model = pytorch_model
            action = choose_action(state, model, input_size)
            step_result = env.step(action)
            next_state, reward, done, _ = step_result[:4]
            
            # Generate adversarial example using the same model used for action selection
            adv_state = generate_adversarial_example(state, model)
            adv_action = choose_action(adv_state, model, input_size)
            
            loss = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
            loss = torch.mean(loss).item()
            total_reward += reward
            steps += 1

            if done:
                accuracy = total_reward / steps
                print(f"Episode {i}, accuracy: {accuracy}")
                accuracies.append(accuracy)
                losses.append(loss)

                # Collect performance metrics
                baseline_metrics["CPU"].append(get_cpu_usage())
                baseline_metrics["GPU"].append(get_gpu_usage())
                baseline_metrics["Memory"].append(get_memory_usage())

                # Generate and evaluate adversarial example using the same model
                adv_state = generate_adversarial_example(state, model)
                adv_action = choose_action(adv_state, model, input_size)
                adv_loss = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
                adv_loss = torch.mean(adv_loss).item()
                total_reward += reward
                steps += 1

                # Collect performance metrics for adversarial example
                attack_metrics["CPU"].append(get_cpu_usage())
                attack_metrics["GPU"].append(get_gpu_usage())
                attack_metrics["Memory"].append(get_memory_usage())

                break

            state = next_state

    # Save evaluation results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Accuracy")
    plt.plot(accuracies)
    plt.xlabel("Episode")
    plt.ylabel("Accuracy")
    plt.subplot(1, 2, 2)
    plt.title("Loss")
    plt.plot(losses)
    plt.xlabel("Episode")
    plt.ylabel("Loss")
    plt.savefig("/root/evaluation_results.png")

# 新增的绘图函数
def plot_metrics(metrics, title, filename):
    plt.figure(figsize=(10, 5))
    for metric, values in metrics.items():
        plt.plot(values, label=metric)
    plt.title(title)
    plt.xlabel("Episode")
    plt.ylabel("Usage")
    plt.legend()
    plt.savefig(f"/root/{filename}.png")

def plot_comparison(baseline_metrics, attack_metrics):
    plot_metrics(baseline_metrics, "Baseline Performance", "baseline_performance")
    plot_metrics(attack_metrics, "Attack Performance", "attack_performance")

# 主程序入口
if __name__ == "__main__":
    input_size = env.observation_space.shape[0]
    evaluate(pytorch_model, env, EPISODES, ATTACK_METHOD, input_size)

    # 绘制性能指标图表
    plot_comparison(baseline_metrics, attack_metrics)
