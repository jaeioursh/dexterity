from shadow import hand
from tqdm import tqdm
from time import time
import numpy as np
from learnmtl import learner
from logger import logger
import cv2

def view(save=True):
    
    folder="data"
    idx=0
    single=0
    
    idx=str(idx)
    log=logger()

    if single:
        test=learner(1,1,-1)
        log.load(folder+"/1_-1_"+idx+".pkl")
    else:
        test=learner(20,20,3)
        log.load(folder+"/20_5_"+idx+".pkl")
    test.set_teams(1)
    params=log.pull("data")
    for vals,p in zip(params,test.data["Agent Populations"]):
        for weights,member in zip(vals,p):
            member.__setstate__(tuple(weights))
    if save:
        env = hand(render_mode="rgb_array")
        frames,R=test.view(env)
        print(max(R),np.argmax(R))
        frame_width,frame_height,depth=frames[0].shape
        out = cv2.VideoWriter('outpy'+str(single)+'.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 30, (frame_width,frame_height))
        Frames=np.array_split(frames,len(R))
        Frames=[x for _,x in sorted(zip(R,Frames),key=lambda x:x[0],reverse=1)]
        R=sorted(R,reverse=1)
        for frame,r,idx in zip(Frames,R,range(len(R))):
            for f in frame:
                f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                # font
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (10, 30)
                fontScale = 1
                color = (255, 255, 255)
                thickness = 2
                txt="Idx: "+str(idx)+"  Reward: "+str(r)
                f = cv2.putText(f, txt, org, font, 
                                fontScale, color, thickness, cv2.LINE_AA)
                out.write(f)
        out.release()
    else:
        env = hand(render_mode="human")
        test.view(env)

if __name__ == "__main__":
    view(1)