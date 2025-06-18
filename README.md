# Background
This project investigates the performance of various artificial intelligence training paradigms—Deep Q-Networks (DQN), Genetic Algorithms (GA), and Behavioral Cloning—within the Atari Boxing environment. The goal is to explore the dichotomy between "nature" (evolutionary methods) and "nurture" (learning from data or experience) in shaping AI behavior. DQN, leveraging reinforcement learning and convolutional neural networks, achieved the highest performance, consistently defeating the in-game opponent with significant margins. GA, simulating evolutionary processes through population-based optimization, showed moderate success but was constrained by computational costs and premature convergence. Behavioral Cloning, which imitates human gameplay through supervised learning, struggled due to limited training data and error propagation from human mistakes. Our analysis suggests that learning-based methods such as DQN are currently more effective in this environment, though evolutionary approaches may improve with better resource allocation and algorithmic diversity. Future work includes expanding to more complex gaming environments and enhancing generalization capabilities across agents.

## Setup
You will need to set up a conda environment first:

conda create -n boxing python=3.11

conda activate boxing

pip install -r requirements.txt

Note: you may need to execute this command in order to make sure ROMs are downloaded in the right place:
AutoROM --accept-license

## Genetic Algorithm Usage
The genetic_algo.py supports the following arguments:

    -- train (places the script in training mode)

    -- eval (places the script in evaluation mode)

    --pop_size <integer> <default=20> (Population size)

    --generations <integer> <default=10> (Number of generations)

    --elite_frac <float> <default=0.2> (Select this percent of top performers for the elite population)

    --mutation_rate <float> <default=0.1> (Rate at which mutation occurs)

    --mutation_strength <float> <default=0.02> (Mutation strength)

    --episodes_per_eval <int> <default=3> (Number of evaluation rounds per genome)

    --model_name <str> <default="ga"> (Name of the model you want to train or evaluate)

    --render <bool> <default=False> (Do you want to display the boxing match?)

    --save_csv <bool> <default=False> (Saves csv of scores over evaluation rounds and saves a plot showing average fitness over generations)

fine_tune_ga.sh allows for a more user friendly method to train a larger number of genetic algorithm models. You can simply modify the hyperparameter values to automatically execute a batch hyperparameter tuning session.

Once a model is trained, either through fine_tune_ga.sh or direct command line args to genetic_algo.py, you can extract performance by looking at the train_log.csv. It should display the model name as well as the average cumulative fitness over generations, giving you a general idea of the model's performance during training. If you want some more indepth knowledge of the model's increasing performance over generations, you can look under the GeneticAlgorithm/plots diectory to view the average fitness score of the population over generations.

In order to evaluate a trained model further, we recommend calling the genetic_algo.py with the following arguments:

    python genetic_algo.py --eval --model_name <specify name> --episodes_per_eval <we recommend 100> --save_csv <True> --render <False unless you want to painfully watch all 100 rounds>

Running an evaluation command in the above format generates a csv file under the GeneticAlgorithm/logs directory that displays the score per evaluation round as well as ouputs the average score.



