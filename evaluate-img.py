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

pytorch_model = PolicyNetwork(input_size, output_size)
pytorch_model.load_state_dict(torch.load("policy_net.pth"))

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

    return adv_state.detach().numpy()[0]

    # 可视化对抗性样本
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original state")
    plt.plot(state)  # Plotting state values, you might need to adjust this based on your state representation
    plt.subplot(1, 2, 2)
    plt.title("Adversarial state")
    plt.plot(adv_state)  # Plotting adversarial state values
    plt.show()

    # 保存对抗性样本图像
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title("Original state")
    plt.plot(state)  # Plotting state values, you might need to adjust this based on your state representation
    plt.savefig("original_state.png")

    plt.subplot(1, 2, 2)
    plt.title("Adversarial state")
    plt.plot(adv_state)  # Plotting adversarial state values
    plt.savefig("adversarial_state.png")


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
            adv_state = generate_adversarial_example(state, model)
            adv_action = choose_action(adv_state, model, input_size)
            loss = model(torch.tensor(state, dtype=torch.float32).view(1, -1))
            loss = torch.mean(loss).item()
            total_reward += reward
            steps += 1

            if action != adv_action:
                print(f"Episode {i}, step {steps}, attack method: {attack_method}")
                print(f"Original state: {state}, action: {action}")
                print(f"Adversarial state: {adv_state}, action: {adv_action}")
                print(f"Loss: {loss}")

                plt.figure(figsize=(10, 5))
                plt.subplot(1, 2, 1)
                plt.title("Original state")
                plt.plot(state)  # Plotting state values, you might need to adjust this based on your state representation
                plt.subplot(1, 2, 2)
                plt.title("Adversarial state")
                plt.plot(adv_state)  # Plotting adversarial state values
                plt.show()

                break

            if done:
                accuracy = total_reward / steps
                print(f"Episode {i}, accuracy: {accuracy}")
                accuracies.append(accuracy)
                losses.append(loss)
                break

            state = next_state

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
    plt.show()

if __name__ == "__main__":
    input_size = env.observation_space.shape[0]
    evaluate(pytorch_model, env, EPISODES, ATTACK_METHOD, input_size)
