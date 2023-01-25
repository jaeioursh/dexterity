from shadow import hand
import tqdm
from time import time
import numpy as np

def test():

    #                   x,y,z ,qw,qx,qy,qz
    start_pos=np.array([0,0,0,0,0,0,0],dtype=np.float64)
    end_pos=np.array([0,0,0,1,0,0,0],dtype=np.float64)
    steps=50
    r_mode="human"
    #r_mode=None
    env = hand(max_episode_steps=steps,render_mode=r_mode)
   
    state, _ = env.reset()
    
    #env.env.env.env.data.randomize_initial_position=False
    #env.env.env.env.data.randomize_initial_rotation=False
    tic=time()
    for q in range(10):
        
        state, _ = env.reset()
        
        for i in range(1):
            state, reward, terminated, truncated, _ = env.step([0 for _ in range(20)])
    toc=time()
    print("Time: ",toc-tic," seconds")

if __name__ == "__main__":
    test()