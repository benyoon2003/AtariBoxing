#!/bin/bash

pop_size=(50)
generations=(80)
elite_frac=(0.2)
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

            model_name="models/GA/ga_ps${ps}_g${g}_ef${ef}_mr${mr}_ms${ms}_epe${epe}.pth"
            
            echo "Training with pop_size=$ps, generations=$g, elite_frac=$ef, mutation_rate=$mr, mutation_strength=$ms, episodes_per_eval=$epe"
            python genetic_algo.py \
              --train \
              --pop_size "$ps" \
              --generations "$g" \
              --elite_frac "$ef" \
              --mutation_rate "$mr" \
              --mutation_strength "$ms" \
              --episodes_per_eval "$epe" \
              --model_path "$model_name"

          done
        done
      done
    done
  done
done
