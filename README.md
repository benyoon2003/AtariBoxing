# Setup
You will need to set up a conda environment first:
conda create -n boxing-dqn python=3.11
conda activate boxing-dqn
pip install -r requirements.txt

# DQN Boxing Agent
To train your agent:
python boxing.py --train

To visualize your agent playing the game:
python boxing.py --eval
