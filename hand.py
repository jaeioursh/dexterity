'''
Provides way to split shadowhand into joints more easily
Each joint has 2 observation dims (angular pos & vel), and one action (angular pos)
The block itself has 13 observation dims, plus 7 observed goal dims, so each
joint maps 22 -> 1
'''
import gymnasium as gym
import numpy as np

class Joint:
    def __init__(self, idxs_obs, idxs_act):
        self.observation_idxs = idxs_obs
        self.action_idxs = idxs_act

    def get_observation(self, obs):
        '''
        Pick own observation out of whole hand's observation vector
        '''
        return obs[self.observation_idxs]

    def ins_action(self, action, act_vec):
        '''
        Insert own action into whole hand's action vector
        I'm not actually using this, but idk maybe one day
        '''
        act_vec[self.action_idxs] = action
        return act_vec


def split_observation(joints, obs):
    obs_list = []

    try:
        iter(obs)
        obs = obs[0]
    except:
        pass
    goal = obs["desired_goal"]

    for j in joints:
        o = obs["observation"][j.observation_idxs]
        o = np.concatenate((o, goal))
        obs_list.append(o)

    return obs_list


def setup_joints():
    '''
    Hardcoded setup to split shadowhand into joints
    '''
    indices_pos = list(range(24))
    indices_vel = list(range(24, 48))
    indices_block = list(range(48,61))

    # Remove non-actuated dof
    for i in [5, 9, 13, 18]:
        indices_pos.remove(i)
    for i in [29, 33, 37, 42]:
        indices_vel.remove(i)

    # Build joints
    joints = []
    for i in range(20):
        idx_pos = indices_pos[i]
        idx_vel = indices_vel[i]

        observation_idxs = [idx_pos, idx_vel]
        observation_idxs.extend(indices_block)
        j = Joint(observation_idxs, i)
        joints.append(j)

    return joints


def test():
    joints = setup_joints()

    env = gym.make("HandManipulateBlockRotateZDense-v1")
    state, _ = env.reset()

    state, reward, terminated, truncated, _ = env.step([0 for _ in range(20)])


if __name__=="__main__":
    test()
