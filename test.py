from shadow import hand
from tqdm import tqdm
from time import time
import numpy as np
from learnmtl import learner
from logger import logger
import multiprocessing as mp

from pettingzoo.sisl import multiwalker_v9

def test(n_agents,learn_type,idx,save,params):

 
    #from guppy import hpy
    #hp = hpy()
    r_mode="human"
    r_mode=None
    #env = [hand(render_mode=r_mode) for i in range(32)]
    env=multiwalker_v9.parallel_env(render_mode=r_mode,n_walkers=n_agents, position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
    terminate_on_fall=False, remove_on_fall=False, terrain_length=200, max_cycles=500)
    #n_agents=20
    #learn_type=3
    test=learner(n_agents,learn_type,params)
    #test=learner(1,1,-1)
    R=[]
    for i in tqdm(range(1000)):
        
        r=test.run(env,0)
        rr=test.test(env)
        test.log.store("reward",rr)
        R.append(r)
        if i%25==24 and save:
            params=[]
            for p in test.data["Agent Populations"]:
                params.append([])
                for member in p:
                    params[-1].append([[np.array(i) for i in member.__getstate__()],member.fitness])
            #print(hp.heap())
            test.log.store("data",params,-1)
            
        
            test.save("data/Q"+str(learn_type)+"_"+str(idx)+".pkl")
    return max(R[-20:])

def action_test():

    render_m="human"
    render_m=None
    env=multiwalker_v9.parallel_env(render_mode=render_m,n_walkers=4, position_noise=1e-3, angle_noise=1e-3, forward_reward=1.0, terminate_reward=-100.0, fall_reward=-10.0, shared_reward=True,
    terminate_on_fall=False, remove_on_fall=False, terrain_length=200, max_cycles=500)
    observations,infos=env.reset(seed=42)[0]
    print(observations)
    R=0.0
    TERM=False
    for i in range(150):
        
    # this is where you would insert your policy
        actions = {agent: env.action_space(agent).sample() for agent in env.agents}
        print(len(actions))
        observations, rewards, terminations, truncations, infos = env.step(actions)
        obs=[observations["walker_"+str(a)] for a in range(4)]
        #R=env.env.unwrapped.env.package.position.x

        if len(env.agents)!=4:
            R-=40
            break
    R+=env.unwrapped.env.package.position.x
    R/=20
    print(R,len(observations))
    print(obs)
   
        
    env.close()
    '''
    env=hand(render_mode="human")
    act=np.array([1 for i in range(20)])*0.6
    env.reset()
    for i in range(100):
        #act*=-1
        env.step(act)
    '''
#train_flag=-1 - Single Agent        
#train_flag=0 - G                        
#train_flag=1 - gapprox
#train_flag=2 - align
#train_flag=3 - g+align
#train_flag=4 - fitness critic

if __name__ == "__main__":
    #action_test()
    '''
    p=[0.001,40.0,60.0,10000.0,0.7]+[53, 0.40, 0.45]
    test(4,0,1000,False,p)
    '''
    #test(20,0,100,parallel=0)
    params=[0.001, 64, 1000]
    for q in [0,1,2,3,4,5]:
        procs=[]
        for i in range(12):
            print(i,q)
            save=True
            idx=i
            learn_type=q
            n_agents=4
            p = mp.Process(target=test, args=(n_agents,learn_type,idx,save,params))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
    '''
    '''
    #    p = mp.Process(target=test, args=(1,-1,i,))
    #    
