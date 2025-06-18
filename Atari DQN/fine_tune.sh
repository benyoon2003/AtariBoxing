#!/bin/bash

batch_sizes=(32) # batch size greater than 32 results in heavy lag
learning_rates=(0.0001 0.00001)
gammas=(0.98 0.995)
epsilons=(1.0)
decay_factors=(0.1 0.2)
episodes_list=(2)

mkdir -p models

for bs in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
      for eps in "${epsilons[@]}"; do
        for decay in "${decay_factors[@]}"; do
          for eps_count in "${episodes_list[@]}"; do

            model_name="Models/dqn_lr${lr}_gamma${gamma}_decay${decay}_850000frames.pth"
            rewards_path="Rewards/dqn_lr${lr}_gamma${gamma}_decay${decay}_850000frames.csv"
            
            echo "Training with lr=$lr, gamma=$gamma, decay=$decay"
            python pixels_train_v2.py \
              --gamma "$gamma" \
              --decay_percentage "$decay" \
              --LR "$lr" \
              --rewards_path "$rewards_path" \
              --model_path "$model_name"

          done
        done
      done
    done
  done
done
