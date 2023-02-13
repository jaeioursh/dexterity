from shadow import hand
from tqdm import tqdm
from time import time
import numpy as np
from learnmtl import learner
from logger import logger
import multiprocessing as mp

def test(n_agents,learn_type,idx,parallel=0):

    
    r_mode="human"
    r_mode=None
    env = [hand(render_mode=r_mode) for i in range(32)]
    #n_agents=20
    #learn_type=3
    test=learner(n_agents,n_agents,learn_type)
    #test=learner(1,1,-1)
    test.set_teams(1)
    for i in tqdm(range(500)):
        if parallel:
            r=test.run(env)
        else:
            r=test.run(env[0],0)
        params=[]
        for p in test.data["Agent Populations"]:
            params.append([])
            for member in p:
                params[-1].append([np.array(i) for i in member.__getstate__()])
            
        test.log.store("data",params,0)
        test.log.store("reward",r)
        if i%25==24:
            test.save("data/"+str(n_agents)+"_"+str(learn_type)+"_"+str(idx)+".pkl")
def action_test():
    env=hand(render_mode="human")
    act=np.array([1 for i in range(20)])*0.6
    env.reset()
    for i in range(100):
        #act*=-1
        env.step(act)

if __name__ == "__main__":
    #test()
    for i in range(6):
        p = mp.Process(target=test, args=(1,-1,i,))
        p.start()