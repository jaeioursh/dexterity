from shadow import hand
from tqdm import tqdm
from time import time
import numpy as np
from learnmtl import learner
from logger import logger
import multiprocessing as mp

def test(n_agents,learn_type,idx,save,params,parallel=0):

 
    #from guppy import hpy
    #hp = hpy()
    r_mode="human"
    r_mode=None
    env = [hand(render_mode=r_mode) for i in range(32)]
    #n_agents=20
    #learn_type=3
    test=learner(n_agents,learn_type,params)
    #test=learner(1,1,-1)
    R=[]
    for i in tqdm(range(700)):
        if parallel:
            r=test.run(env)
        else:
            r=test.run(env[0],0)
        R.append(r)
        if i%25==24 and save:
            params=[]
            for p in test.data["Agent Populations"]:
                params.append([])
                for member in p:
                    params[-1].append([[np.array(i) for i in member.__getstate__()],member.fitness])
            #print(hp.heap())
            test.log.store("data",params,-1)
            test.log.store("reward",r)
        
            test.save("data/"+str(learn_type)+"_"+str(idx)+".pkl")
    return max(R[-20:])
def action_test():
    env=hand(render_mode="human")
    act=np.array([1 for i in range(20)])*0.6
    env.reset()
    for i in range(100):
        #act*=-1
        env.step(act)
#train_flag=-1 - Single Agent        
#train_flag=0 - G                        
#train_flag=1 - gapprox
#train_flag=2 - align
#train_flag=3 - g+align
#train_flag=4 - fitness critic

if __name__ == "__main__":
    test(20,0,100,parallel=0)
    for q in [0,1,2,3,4]:
        procs=[]
        for i in range(12):
            print(i,q)
            p = mp.Process(target=test, args=(20,q,i,))
            p.start()
            procs.append(p)
        for p in procs:
            p.join()
        
    #    p = mp.Process(target=test, args=(1,-1,i,))
    #    
