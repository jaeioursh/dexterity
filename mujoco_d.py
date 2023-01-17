'''
Evaluate difference rewards for mujoco-based gym environments
'''
import gymnasium as gym
import numpy as np
from copy import deepcopy


def sim_copy(source, target):
    '''
    Copy the mujoco internals from one environment to another
    NOTE: just doing target.data=deepcopy(source.data) does
    NOT work (results will not be identical after 1 step) and
    I don't know why, hence this nasty operation
    @param source: The environment to replicate
    @param target: The environment to copy source
    '''
    # Np random is used for some of the solvers
    target.np_random = deepcopy(source.np_random)

    # Copy all writeable properties from source.data
    # This list is updated as of gymnasium-robotics 1.2.0,
    # but the more general, slower solution of automatically
    # checking all attributes is retained for posterity.
    data_attr = ['D_colind', 'D_rowadr', 'D_rownnz', 'act', 'act_dot', 'actuator_force', 'actuator_length', 'actuator_moment', 'actuator_velocity', 'cacc', 'cam_xmat', 'cam_xpos', 'cdof', 'cdof_dot', 'cfrc_ext', 'cfrc_int', 'cinert', 'crb', 'ctrl', 'cvel', 'energy', 'geom_xmat', 'geom_xpos', 'light_xdir', 'light_xpos', 'maxuse_con', 'maxuse_efc', 'maxuse_stack', 'mocap_pos', 'mocap_quat', 'nbuffer', 'ncon', 'ne', 'nefc', 'nf', 'nstack', 'plugin', 'plugin_data', 'plugin_state', 'pstack', 'qDeriv', 'qH', 'qHDiagInv', 'qLD', 'qLDiagInv', 'qLDiagSqrtInv', 'qLU', 'qM', 'qacc', 'qacc_smooth', 'qacc_warmstart', 'qfrc_actuator', 'qfrc_applied', 'qfrc_bias', 'qfrc_constraint', 'qfrc_inverse', 'qfrc_passive', 'qfrc_smooth', 'qpos', 'qvel', 'sensordata', 'site_xmat', 'site_xpos', 'solver_fwdinv', 'solver_iter', 'solver_nnz', 'subtree_angmom', 'subtree_com', 'subtree_linvel', 'ten_J', 'ten_J_colind', 'ten_J_rowadr', 'ten_J_rownnz', 'ten_length', 'ten_velocity', 'ten_wrapadr', 'ten_wrapnum', 'time', 'userdata', 'wrap_obj', 'wrap_xpos', 'xanchor', 'xaxis', 'xfrc_applied', 'ximat', 'xipos', 'xmat', 'xpos', 'xquat']

    for attr in data_attr:
        value = deepcopy(source.data.__getattribute__(attr))
        target.data.__setattr__(attr, value)

    # Automatically try all attributes
    #     data_attr = [x for x in dir(env.data) if not x.startswith("_")]
    #
    #    for attr in data_attr:
    #        try:
    #            value = deepcopy(source.data.__getattribute__(attr))
    #            target.data.__setattr__(attr, value)
    #        except AttributeError as e:
    #            print(attr, e)
    #        except TypeError as e:
    #            print(attr, e)


def rand_action():
    return np.random.random((20,))


def zero_action():
    return np.zeros((20,))


def rand_actions(n=10):
    l = []
    for i in range(n):
        l.append(rand_action())
    return l


def mujoco_d(env, action, agent_action_idxs, env2=None):
    '''
    Take an action in the environment and calculate D based
    on a zero counterfactual
    @param env: The environment to be tested in
    @param action: The JOINT action taken
    @param agent_action_idxs: List of lists, each item contains
        the indices corresponding to an agent's contribution to
        the joint action
    @param env2: Optional, an environment to use to evaluate
        counterfactuals so that one doesn't need to be created
        every time this function is called.
    @return state: observed state after taking action
    @return G: reward achieved by the full joint-action
    @return D: list containing D for each agent
    @return done: truncated or terminated
    '''
    cf_G = []

    if env2 is None:
        env2 = gym.make(env.unwrapped.spec.id)
        env2.reset()

    # FIRST, calculate rewards for all counterfactuals
    for action_idxs in agent_action_idxs:
        # Generate counterfactual action
        cf_action = deepcopy(action)
        cf_action[action_idxs] = 0

        # Copy and score counterfactual
        sim_copy(env, env2)
        _, d, _, _, _ = env2.step(cf_action)
        cf_G.append(d)

    # THEN, run actual action to get G
    state, G, trun, term, _ = env.step(action)
    done = trun or term

    # D = G(z) - G(z_-i U cf)
    cf_G = np.array(cf_G)
    D = G - cf_G

    return state, G, D, done


def test_copy():
    env1 = gym.make("HandReachDense-v1")
    env2 = gym.make("HandReachDense-v1")
    env1.reset()
    env2.reset()

    # Perform a sequence of actions on 1
    for a in rand_actions():
        env1.step(a)

    # Copy state into 2
    sim_copy(env1, env2)

    # Take the same action on both envs
    a = rand_action()
    state1, _, _, _, _ = env1.step(a)
    state2, _, _, _, _ = env2.step(a)

    # Make sure they ended up on the same state
    diff = state1["observation"] - state2["observation"]
    sse = np.sum(np.power(diff, 2))
    if sse < 1e-5:
        print(f"Test passed, sse of {sse:E}")
    else:
        raise Exception(f"TEST FAILED, sse of {sse:E}")


def test_d():
    env = gym.make("HandReachDense-v1")
    env.reset()

    # Start with some random sequence
    for a in rand_actions():
        env.step(a)

    # Try to calculate D assuming each index is an agent
    a = rand_action()
    agent_action_idxs = [[i] for i in range(20)]
    state, G, D, done = mujoco_d(env, a, agent_action_idxs)

    print("Test passed, calculated D:")
    for i, d in enumerate(D):
        print(f"  agent {i} = {d:.3E}")


if __name__=="__main__":
    test_copy()
    test_d()
