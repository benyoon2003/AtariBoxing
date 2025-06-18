#!/bin/bash

# Modify these parameters to your liking to perform grid search hyperparameter tuning
pop_size=(50 30 10)
generations=(80)
elite_frac=(0.2 0.1)
mutation_rate=(0.1)
mutation_strength=(0.02)
episodes_per_eval=(3)

mkdir -p models

for ps in "${pop_size[@]}"; do
  for g in "${generations[@]}"; do
    for ef in "${elite_frac[@]}"; do
      for mr in "${mutation_rate[@]}"; do
        for ms in "${mutation_strength[@]}"; do
          for epe in "${episodes_per_eval[@]}"; do

            model_name="ga_ps${ps}_g${g}_ef${ef}_mr${mr}_ms${ms}_epe${epe}"
            
            echo "Training with pop_size=$ps, generations=$g, elite_frac=$ef, mutation_rate=$mr, mutation_strength=$ms, episodes_per_eval=$epe"
            python genetic_algo.py \
              --train \
              --pop_size "$ps" \
              --generations "$g" \
              --elite_frac "$ef" \
              --mutation_rate "$mr" \
              --mutation_strength "$ms" \
              --episodes_per_eval "$epe" \
              --model_name "$model_name"

          done
        done
      done
    done
  done
done
