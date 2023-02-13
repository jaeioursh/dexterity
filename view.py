from shadow import hand
from tqdm import tqdm
from time import time
import numpy as np
from learnmtl import learner
from logger import logger
import cv2

def view(save=True):
    test=learner(1,1,-1)
    test.set_teams(1)
    log=logger()
    log.load("data/1_-1_0.pkl")
    params=log.pull("data")[-1][-1]
    for vals,p in zip(params,test.data["Agent Populations"]):
        for weights,member in zip(vals,p):
            member.__setstate__(tuple(weights))
    if save:
        env = hand(render_mode="rgb_array")
        frames=test.view(env)
        frame_width,frame_height,depth=frames[0].shape
        out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 60, (frame_width,frame_height))
        for f in frames:
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            out.write(f)
        out.release()
    else:
        env = hand(render_mode="human")
        test.view(env)

if __name__ == "__main__":
    view(0)