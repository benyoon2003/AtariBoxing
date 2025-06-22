import gymnasium as gym
import ale_py
import os
import numpy as np
import h5py

from gymnasium.utils.play import play

action_list = []
num_obs = 0

# creating new file name for actions

def generate_Action_Name():
    action_file_name = "actions/actions_0"
    created_new_file = False
    num_loops = 0
    while not created_new_file:
        print("Creating new File")
        temp_file_path = os.path.join("/Users/andrew/Documents/CS4100/", action_file_name + ".npy")
        print(temp_file_path)
        if not os.path.isfile(temp_file_path):
            print(os.path.isfile(temp_file_path))
            break
        
        action_file_name = "actions/actions_" + str(num_loops)
        num_loops += 1

    return action_file_name

#creating new directory for observations

def generate_Observations_Name():
    observation_file_name = "observations/observations_0/"
    created_new_file = False
    num_loops = 0
    while not created_new_file:
        print("Creating new File")
        temp_file_path = os.path.join("/Users/andrew/Documents/CS4100/", observation_file_name)
        print(temp_file_path)
        if not os.path.isdir(temp_file_path):
            print(os.path.isfile(temp_file_path))
            break
        
        observation_file_name = "observations/observations_" + str(num_loops)
        num_loops += 1

    return observation_file_name

action_file_name = generate_Action_Name()
observation_dir_name = generate_Observations_Name()

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    if not type(obs_t) is tuple and not type(obs_t) is dict:
        
        global action_list, num_obs, action_file_name, observation_dir_name
        try:
            os.mkdir(observation_dir_name)
        except FileExistsError:
            pass #intended behavior here but still kinda stupid
        action_list.append(action)
        obs_file_name = observation_dir_name + "/obs_" + str(num_obs)
        np.save(obs_file_name, obs_t)
        np.save(action_file_name, action_list)

        num_obs += 1

def main():

    if not os.path.isdir(os.getcwd() + "/actions"):
        os.mkdir("actions")
    if not os.path.isdir(os.getcwd() + "/observations"):
        os.mkdir("observations")

    env = gym.make("ALE/Boxing-v5", render_mode="rgb_array", obs_type="grayscale")
    env.reset()
    play(env, keys_to_action={
    " ": 1,
    "w": 2,
    "d": 3,
    "a": 4,
    "s": 5,
    "wd": 6,
    "wa": 7,
    "sd": 8,
    "sa": 9,
    "w ": 10,
    "d ": 11,
    "a ": 12,
    "s ": 13,
    "wd ": 14,
    "wa ": 15,
    "sd ": 16,
    "as ": 17,
}, noop=0, callback=callback, fps=30)


if __name__ == "__main__":
    main()
