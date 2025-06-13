#!/bin/bash 

batch_size=16 # batch size greater than 32 results in heavy lag
learning_rate=0.0001
gamma=0.98
epsilon=1.0
decay_factor=0.99
episodes_list=(150 250 350 500)

mkdir -p models


for eps_count in "${episodes_list[@]}"; do

model_name="models/DQN/BENCHMARK_dqn_bs${batch_size}_lr${learning_rate}_gamma${gamma}_eps${epsilon}_decay${decay_factor}_ep${eps_count}.pth"

echo "Training with batch_size=$batch_size, lr=$learning_rate, gamma=$gamma, epsilon=$epsilon, decay=$decay_factor, episodes=$eps_count"
python dqn.py \
    --train \
    --batch_size "$batch_size" \
    --gamma "$gamma" \
    --epsilon "$epsilon" \
    --decay_factor "$decay_factor" \
    --LR "$learning_rate" \
    --episodes "$eps_count" \
    --model_path "$model_name"

done

pop_size=(20 40 60)
generations=(10 30 50)
elite_frac=0.2
mutation_rate=0.1
mutation_strength=0.02
episodes_per_eval=3

mkdir -p models

for ps in "${pop_size[@]}"; do
  for g in "${generations[@]}"; do
    model_name="models/GA/ga_ps${ps}_g${g}_ef${elite_frac}_mr${mutation_rate}_ms${mutation_strength}_epe${episodes_per_eval}.pth"
    
    echo "Training with pop_size=$ps, generations=$g, elite_frac=$elite_frac, mutation_rate=$mutation_rate, mutation_strength=$mutation_strength, episodes_per_eval=$episodes_per_eval"
    python genetic_algo.py \
        --train \
        --pop_size "$ps" \
        --generations "$g" \
        --elite_frac "$elite_frac" \
        --mutation_rate "$mutation_rate" \
        --mutation_strength "$mutation_strength" \
        --episodes_per_eval "$episodes_per_eval" \
        --model_path "$model_name"

  done
done