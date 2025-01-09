import gymnasium as gym
import random
import numpy as np
import ale_py
gym.register_envs(ale_py)
from gym.utils.play import play
import sys
import os


from ale_py import ALEInterface

import time
import imageio
ale = ALEInterface()

global_time_folder_second = str(int(time.time()))
path_ext = "lab5/data/"+global_time_folder_second+"/"
os.makedirs(path_ext, exist_ok=True)

ale.loadROM("pong.bin")

previmg = None

def clbk(obs_t, img, action, rew, done, info, other):
    print(obs_t[0].shape, action, rew, done, info, other)
    file_name = f"pong_{other['frame_number']}.png"
    if previmg is not None:
        imageio.imwrite(path_ext+file_name, img[34:194:4,12:148:2,1]-previmg)
    previmg = img[34:194:4,12:148:2,1]

    time.sleep(0.1)
    with open(path_ext+"dataset.csv", "a", encoding="utf-8") as f:
        f.write(f"{file_name},{action},{rew}\n")
    if done:
        sys.exit(0)

    pass

# Initialise the environment
env = gym.make("ALE/Pong-v5", render_mode="rgb_array")


env.reset()
play(env, zoom=5, callback=clbk)
