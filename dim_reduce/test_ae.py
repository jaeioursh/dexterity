'''
Test whether the autoencoder has learned anything meaningful
by giving it a good looksie
'''
import time
import torch
import pickle
import numpy as np
import gymnasium as gym
from generate_data import Walker
from autoencoder import Autoencoder, normalize, unnormalize


def main():
    # Set up environment
    env = gym.make("HandReach-v1", render_mode="human")
    env = gym.make("HandReach-v1", render_mode="human")
    env.reset()

    # Set up model
    with open("ae_trained_48_100_24_6.pkl", "rb") as f:
        savedstuff = pickle.load(f)
    model = savedstuff["model"]

    # Flattening to 1d makes everything easier later
    mins = savedstuff["mins"].flatten()
    ranges = savedstuff["ranges"].flatten()

    for i in range(20):
        # Get to some random state
        w = Walker()
        for i in range(20):
            state, _, _, _, _ = env.step(w.step())

        # Turn observation into network input
        x = state["observation"][0:48]
        x = normalize(x, mins, ranges)
        x_test = unnormalize(x, mins, ranges)
        x = torch.Tensor(x)

        # Infer with model
        with torch.no_grad():
            out = model.forward(x).detach().numpy()

        # Unpackage joint positions and velocities
        out = unnormalize(out, mins, ranges)
        pos = out[:24]
        vel = out[24:]

        # Jam parameters into environment and render
        print("The autoencoder predicted this:")
        time.sleep(1)
        env.data.qpos = pos
        env.data.qvel = vel
        env.render()
        time.sleep(1)

    # And then there's a weird exception on __del__ for some reason


if __name__=="__main__":
    main()
