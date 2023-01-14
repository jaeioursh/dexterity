import gymnasium as gym
from learnmtl import learner


def main():
    env = gym.make("HandManipulateBlockRotateZDense-v1")

    l = learner(20, 19)
    l.set_teams(20)
    l.run(env, 3)


if __name__=="__main__":
    main()
