#!/bin/bash

# Corrected the shebang line
# Corrected the use of arrays and loops in bash

source_model_path=("policy_net_dqn1.pth" "policy_net_ddqn1.pth")
victim_model_paths=("policy_net_dqn2.pth" "policy_net_ddqn2.pth")
epsilons=(0.001)  # Removed commas
methods=("random" "fgsm" "mifgsm" "nifgsm")

for source_model_path in "${source_model_path[@]}"; do
  for target_model_path in "${victim_model_paths[@]}"; do
    for epsilon in "${epsilons[@]}"; do
      for method in "${methods[@]}"; do
        python black_box_attack.py --source_model_path "$source_model_path" --target_model_path "$target_model_path" --epsilon "$epsilon" --method "$method"
      done
    done
  done
done