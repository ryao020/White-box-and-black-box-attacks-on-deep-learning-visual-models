import torch
import numpy as np
import gym
from torch import nn
import argparse

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


# Function to generate adversarial state
def generate_adversarial_state(state, epsilon, method="fgsm"):
    if method == "fgsm":
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        state.requires_grad = True

        output = model(state)
        loss = -torch.max(output)
        loss.backward()

        state_grad = state.grad.data
        adversarial_state = state + epsilon * state_grad.sign()
        adversarial_state = adversarial_state.detach()

        return adversarial_state.cpu().numpy().squeeze()

    elif method == "mifgsm":
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        state.requires_grad = True

        for i in range(10):
            output = model(state)
            loss = -torch.max(output)
            loss.backward()

            state_grad = state.grad.data
            state = state + epsilon * state_grad.sign()
            state = state.detach()
            state.requires_grad = True

        return state.cpu().detach().numpy().squeeze()

    elif method == "nifgsm":
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0).to(device)
        state.requires_grad = True

        for i in range(10):
            output = model(state)
            loss = -torch.max(output)
            loss.backward()

            state_grad = state.grad.data
            state = state + epsilon * state_grad.sign()
            state = state.detach()
            state.requires_grad = True

        return state.cpu().detach().numpy().squeeze()

    elif method == "random":
        return state + np.random.uniform(-epsilon, epsilon, state.shape)


def normalize_state(state):
    # assume state[1] from [-1e8, 1e8] and state[3] from [-1e8, 1e8]
    # state[0] from [-2.4, 2.4] and state[2] from [-0.418, 0.418]

    # normalize the state to [0,1] along each dimension
    state[0] = (state[0] + 2.4) / 4.8
    state[1] = (state[1] + 1e8) / 2e8
    state[2] = (state[2] + 0.418) / 0.836
    state[3] = (state[3] + 1e8) / 2e8
    return state

def denormalize_state(state):
    state[0] = state[0] * 4.8 - 2.4
    state[1] = state[1] * 2e8 - 1e8
    state[2] = state[2] * 0.836 - 0.418
    state[3] = state[3] * 2e8 - 1e8
    return state



# Test the model with adversarial attacks in the environment
def test_with_adversarial_attacks(steps=100, epsilon=0.1, args=None):
    rewards = []
    print(f"Testing the model with adversarial attacks using {args.method} method, epsilon: {epsilon}...")

    for trail in range(args.trails_number):

        total_reward = 0
        state = env.reset()
        model.eval()

        for step in range(steps):
            # env.render()

            # need to normalize the state to [0,1] along each dimension before attacking

            state = normalize_state(state)
            state = generate_adversarial_state(state, epsilon, method=args.method)
            state = denormalize_state(state)

            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = model(state_tensor).max(1)[1].view(1, 1).item()

            state, reward, done, _ = env.step(action)
            total_reward += reward

            if done:
                break

        rewards.append(total_reward)
        # print(f"Total Reward with adversarial attacks: {total_reward}")

    print("Rewards: ", rewards)
    print(f"Average Reward with adversarial attacks: {np.mean(rewards)}")
    print(f"Standard deviation: {np.std(rewards)}")
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--epsilon", type=float, default=0.01)
    parser.add_argument("--model_path", type=str, default="policy_net_dqn1.pth")
    parser.add_argument("--method", type=str, default="random", choices=[ "random", "fgsm", "mifgsm", "nifgsm"])
    parser.add_argument("--trails_number", type=int, default=10)
    args = parser.parse_args()

    # Environment setup
    env = gym.make('CartPole-v1')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("model_path: ", args.model_path)

    # Load the trained model
    model = DQN(state_dim, action_dim).to(device)
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    env = gym.make('CartPole-v1')

    test_with_adversarial_attacks(steps=args.steps, epsilon=args.epsilon, args=args)
