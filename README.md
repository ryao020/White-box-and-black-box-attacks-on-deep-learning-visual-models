

```markdown
# Deep Reinforcement Learning Adversarial Attacks

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains code for training a Deep Reinforcement Learning (DRL) agent and implementing white-box adversarial attacks using Fast Gradient Sign Method (FGSM), Momentum Iterative FGSM (MI-FGSM), and Noise Injection FGSM (NI-FGSM). The trained agent is tested in the CartPole-v1 environment from OpenAI Gym.

## Table of Contents
- [About](#about)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Training the Agent](#training-the-agent)
  - [Running Adversarial Attacks](#running-adversarial-attacks)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## About

This project aims to explore adversarial attacks on DRL agents and assess the transferability of these attacks. The implemented attacks include FGSM, MI-FGSM, and NI-FGSM, and the agent is trained in the CartPole-v1 environment. The README provides guidance on setting up the project, training the agent, and running adversarial attacks.

## Getting Started

### Prerequisites

- Python 3.7 or higher
- PyTorch
- Gym

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/your-repo.git
   cd your-repo
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

### Training the Agent

Run the following command to train the DRL agent:

```bash
python3 train_agent.py
```

### Running Adversarial Attacks

Modify the `attack.py` script as needed and execute it:

```bash
python3 attack.py
```

## Results

The results of adversarial attacks for different epsilon values are as follows:

- Epsilon: 0.01
  - Normal Reward: 18.12, Normal Rate: 0.87
  - Adversarial Reward: 19.11, Adversarial Rate: 0.94

- Epsilon: 0.02
  - Normal Reward: 19.35, Normal Rate: 0.88
  - Adversarial Reward: 18.21, Adversarial Rate: 0.87

... (results for other epsilon values)

## Contributing

Contributions are welcome! Please follow the [contribution guidelines](CONTRIBUTING.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgements

- [OpenAI Gym](https://gym.openai.com/)
- [PyTorch](https://pytorch.org/)
```
