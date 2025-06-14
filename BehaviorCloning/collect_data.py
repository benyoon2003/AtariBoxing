import gymnasium as gym
import h5py

from gymnasium.utils.play import play

def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    print(obs_t)

def main():


    env = gym.make("ALE/Boxing-v5", render_mode="rgb_array", obs_type="grayscale")
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