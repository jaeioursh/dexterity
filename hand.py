'''
Provides way to split shadowhand into joints more easily
Each joint has 2 observation dims (angular pos & vel), and one action (angular pos)
The block itself has 13 observation dims, plus 7 observed goal dims, so each
joint maps 22 -> 1
'''
import gymnasium as gym
import numpy as np
from time import time

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
    #                   x,y,z ,qw,qx,qy,qz
    start_pos=np.array([0,0,0,0,0,0,0],dtype=np.float64)
    end_pos=np.array([0,0,0,1,0,0,0],dtype=np.float64)
    steps=50
    r_mode="human"
    #r_mode=None
    env = gym.make("HandManipulateBlockRotateZDense-v1",target_position="fixed",target_rotation="fixed",max_episode_steps=steps,render_mode=r_mode)
    #env.env.env.env.randomize_initial_position=False
    #env.env.env.env.randomize_initial_rotation=False
    state, _ = env.reset()
    #end_pos=env.env.env.env.unwrapped.goal.copy()
    #env.env.env.env.data.randomize_initial_position=False
    #env.env.env.env.data.randomize_initial_rotation=False
    tic=time()
    for q in range(10):
        
        state, _ = env.reset()
        #env.env.env.env.goal=end_pos
        obs=env.env.env.env._get_obs()
        diff=obs["observation"]-state["observation"]
        print(obs)
        for i in range(1):
            state, reward, terminated, truncated, _ = env.step([0 for _ in range(20)])
            
    toc=time()
    print("Time: ",toc-tic," seconds")

if __name__=="__main__":
    test()
