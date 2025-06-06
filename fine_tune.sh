#!/bin/bash

batch_sizes=(32 64)
learning_rates=(0.00005 0.0001 0.0000)
gammas=(0.98 0.99 0.995)
epsilons=(1.0)
decay_factors=(0.99 0.995)
episodes_list=(100)

mkdir -p models

for bs in "${batch_sizes[@]}"; do
  for lr in "${learning_rates[@]}"; do
    for gamma in "${gammas[@]}"; do
      for eps in "${epsilons[@]}"; do
        for decay in "${decay_factors[@]}"; do
          for eps_count in "${episodes_list[@]}"; do

            model_name="models/dqn_bs${bs}_lr${lr}_gamma${gamma}_eps${eps}_decay${decay}_ep${eps_count}.pth"
            
            echo "Training with batch_size=$bs, lr=$lr, gamma=$gamma, epsilon=$eps, decay=$decay, episodes=$eps_count"
            python boxing.py \
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
