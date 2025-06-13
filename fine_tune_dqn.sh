#!/bin/bash

batch_sizes=(32) # batch size greater than 32 results in heavy lag
learning_rates=(0.0001 0.00001)
gammas=(0.98 0.995)
epsilons=(1.0)
decay_factors=(0.99 0.995)
episodes_list=(1000)

mkdir -p models

for bs in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
      for eps in "${epsilons[@]}"; do
        for decay in "${decay_factors[@]}"; do
          for eps_count in "${episodes_list[@]}"; do

            model_name="models/DQN/dqn_bs${bs}_lr${lr}_gamma${gamma}_eps${eps}_decay${decay}_ep${eps_count}.pth"
            
            echo "Training with batch_size=$bs, lr=$lr, gamma=$gamma, epsilon=$eps, decay=$decay, episodes=$eps_count"
            python dqn.py \
              --train \
              --batch_size "$bs" \
              --gamma "$gamma" \
              --epsilon "$eps" \
              --decay_factor "$decay" \
              --LR "$lr" \
              --episodes "$eps_count" \
              --model_path "$model_name"

          done
        done
      done
    done
  done
done
