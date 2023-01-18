import gymnasium as gym
import tqdm
from learnmtl import learner


def main():

    env = gym.make("HandManipulateBlockRotateZDense-v1", max_episode_steps=25)
            #sub team size, total number of agents
    l = learner(20, 20)
    l.set_teams(1)

    pbar = tqdm.trange(100)
    for i in pbar:
        pbar.set_description(f"{i}: {l.run(env, 3)}")


if __name__=="__main__":
    main()
