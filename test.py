from shadow import hand
from tqdm import tqdm
from time import time
import numpy as np
from learnmtl import learner
from logger import logger
def test():

    
    r_mode="human"
    r_mode=None
    env = [hand(render_mode=r_mode) for i in range(32)]
    #test=learner(20,20,3)
    test=learner(1,1,-1)
    test.set_teams(1)
    for i in tqdm(range(500)):
        print(test.run(env))
    params=[]
    for p in test.data["Agent Populations"]:
        params.append([])
        for member in p:
            params[-1].append(member.__getstate__())
        
        test.put("data",params)
        test.save()

if __name__ == "__main__":
    test()
    