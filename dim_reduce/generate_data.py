import gymnasium as gym
import numpy as np
import tqdm
import pickle


class Walker:
    '''
    Gaussian normal walks in n dimensions, bounded [-1, 1]
    '''
    def __init__(self, n_dims=20, lower=-1.0, upper=1.0, stepstd=1.0):
        assert upper > lower

        self.n_dims = n_dims
        self.lb = lower
        self.ub = upper
        self.stepstd = stepstd

        self.reset()


    def reset(self):
        range = self.ub - self.lb
        mid = self.lb + (range / 2)
        self.state = range * np.random.random((self.n_dims,)) - mid


    def step(self):
        diff = self.stepstd * np.random.standard_normal((self.n_dims,))
        self.state = np.clip(self.state + diff, self.lb, self.ub)
        return self.state


def walk_hand(n=10000):
    env = gym.make("HandReachDense-v1")
    env.reset()
    w = Walker()

    # How many observed dims
    obs_dims = 48
    ret = np.zeros((obs_dims, n))

    for i in tqdm.trange(n):
        if i % 100 == 0:
            w.reset()
        a = w.step()
        state, _, _, _, _ = env.step(a)
        ret[:,i] = state["observation"][:obs_dims]

    return ret


def main():
    '''
    Runs lots and lots of experiences through the gym and saves the observations
    '''
    x = walk_hand(100000)
    with open("hand_data.pkl", "wb") as f:
        pickle.dump(x, f)


if __name__=="__main__":
    main()
