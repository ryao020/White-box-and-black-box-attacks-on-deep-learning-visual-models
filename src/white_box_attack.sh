#!/bin/bash

# Corrected the shebang line
# Corrected the use of arrays and loops in bash

model_paths=("policy_net_dqn1.pth" "policy_net_dqn2.pth" "policy_net_ddqn1.pth" "policy_net_ddqn2.pth")
epsilons=(0.0001 0.001 0.01 0.1 1.0)  # Removed commas
methods=("fgsm" "mifgsm" "nifgsm")

for model_path in "${model_paths[@]}"; do
  for epsilon in "${epsilons[@]}"; do
    for method in "${methods[@]}"; do
      python white_box_attack.py --model_path "$model_path" --epsilon "$epsilon" --method "$method"
    done
  done
done