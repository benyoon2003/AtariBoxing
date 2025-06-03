import gymnasium as gym

from gymnasium.utils.play import play

env = gym.make("ALE/Boxing-v5", render_mode="rgb_array")
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
}, noop=0)