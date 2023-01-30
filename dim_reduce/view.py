import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import gymnasium as gym
from generate_data import Walker
from autoencoder import Autoencoder, normalize, unnormalize
from math import sqrt,ceil, isnan
from multiprocessing import Process,Pipe
import sys



def get_view(conn):
    global env
    while True:
        out=conn.recv()
        pos = out[:24]
        vel = out[24:]
        env.data.qpos = pos
        env.data.qvel = vel
        img=env.render()
        conn.send(img)

def clear_var(event):
    global clicked
    clicked=False

def set_var(event):
    global clicked
    clicked=True    

def onclick(event,axs,model,mins,ranges,wndw,conn):
    idx=0
    global clicked

    
    for i in range(len(axs)-1):
        if axs[i]==event.inaxes:
            idx=i
    global x
    global handle
    if event.xdata is None or event.ydata is None or not clicked:
        return

    x[idx*2]  =event.xdata
    x[idx*2+1]=event.ydata
    with torch.no_grad():
        out = model.decoder.forward(torch.Tensor(x)).detach().numpy()
    
    # Unpackage joint positions and velocities
    out = unnormalize(out, mins, ranges)
    conn.send(out)

    img=conn.recv()
    print(handle)
    if handle is None:
        
        handle=wndw.imshow(img)
    else:
        handle.set_data(img)
        
    wndw.set_title(np.array2string(np.round(x,2)))
    plt.draw()
def get_data(model,mins,ranges):
    with open("hand_data.pkl", "rb") as f:
            data = pickle.load(f)
    
    data_norm = normalize(data, mins, ranges)
    return model.encoder.forward(torch.Tensor(data_norm.T)).detach().numpy()


def view(fname,conn):

    with open(fname, "rb") as f:
        savedstuff = pickle.load(f)
    model = savedstuff["model"]
    out_size=savedstuff["architecture"][-1]


    # Flattening to 1d makes everything easier later
    mins = savedstuff["mins"]
    ranges = savedstuff["ranges"]
    y=get_data(model,np.array([mins]).T,np.array([ranges]).T)
    y=y[::20,:]
    color=np.linalg.norm(y,axis=1)
    dims=ceil(sqrt(out_size/2+1))
    fig,axs=plt.subplots(dims,dims)
    if dims==1:
        axs=[axs]
    else:
        axs=axs.flatten()
    for i in range(out_size//2):
        axs[i].scatter(y[:,i*2],y[:,i*2+1],c=color,s=20)
        #axs[i].set_xlim(0,1)
        #axs[i].set_ylim(0,1)
    
    global x
    x=np.zeros(out_size)
    
    fig.canvas.mpl_connect("motion_notify_event", lambda event: onclick(event,axs,model,mins,ranges,axs[-1],conn))
    fig.canvas.mpl_connect("button_press_event", set_var)
    fig.canvas.mpl_connect("button_release_event", clear_var)
    plt.pause(0)
if __name__ =="__main__":
    x=[]
    handle=None
    clicked=False
    env = gym.make("HandReach-v1", render_mode="rgb_array")
    env.reset()
    rec,snd=Pipe()

    #fname="ae_trained_48_100_24_6.pkl"
    fname=sys.argv[1]
    
    Process(target=view,args=(fname,rec,)).start()
    Process(target=get_view,args=(snd,)).start()
    #view()