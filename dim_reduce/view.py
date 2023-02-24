import matplotlib
#matplotlib.use('TkAgg')

import matplotlib.pyplot as plt
import numpy as np
import torch
import pickle
import gymnasium as gym
from autoencoder import Autoencoder, normalize, unnormalize
from math import sqrt,ceil, isnan
from multiprocessing import Process,Pipe
import sys
import io
from sklearn.cluster import KMeans

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

def get_view(conn):
    global env
    while True:
        out=conn.recv()
        pos = out[:24]
        vel = out[24:]
        env.data.qpos[:] = 100
        env.data.qvel[:] = 100
        env.data.qpos[:24] = pos
        env.data.qvel[:24] = vel
        img=env.render()
        conn.send(img)

def clear_var(event):
    global clicked
    clicked=False

def set_var(event):
    global clicked
    clicked=True    

def onclick(event,axs,model,mins,ranges,wndw,conn,idxs):
    idx=0
    global clicked

    
    for i in range(len(axs)-1):
        if axs[i]==event.inaxes:
            idx=i
    global x
    global handle
    if event.xdata is None or event.ydata is None or not clicked:
        return

    x[idxs[idx*2]]  =event.xdata
    x[idxs[idx*2+1]]=event.ydata
    with torch.no_grad():
        out = model.decoder.forward(torch.Tensor(x)).detach().numpy()
    
    # Unpackage joint positions and velocities
    out = unnormalize(out, mins, ranges)
    conn.send(out)

    img=conn.recv()

    if handle is None:
        
        handle=wndw.imshow(img)
    else:
        handle.set_data(img)
        
    wndw.set_title(np.array2string(np.round(x,2)))
    plt.draw()
def get_data(model,mins,ranges):
    with open("hand_data.pkl", "rb") as f:
            data = pickle.load(f)
    #data=data[:24,:]
    data_norm = normalize(data, mins, ranges)
    return model.encoder.forward(torch.Tensor(data_norm.T)).detach().numpy()

def reindex(c):
    for i in range(len(c)):
        c[i,i]=0
    c=np.abs(c)
    idxs=[]
    for i in range(len(c)//2):
        j,k = np.unravel_index(np.argmax(c, axis=None), c.shape)
        idxs+=[j,k]
        c[j,:]=0
        c[:,j]=0
        c[k,:]=0
        c[:,k]=0
    return idxs

def reindex2(c):
    pass

def view(fname,conn):

    with open(fname, "rb") as f:
    #    savedstuff = pickle.load(f)
        savedstuff=CPU_Unpickler(f).load()
    model = savedstuff["model"]
    out_size=savedstuff["architecture"][-1]


    # Flattening to 1d makes everything easier later
    mins = savedstuff["mins"]
    ranges = savedstuff["ranges"]
    y=get_data(model,np.array([mins]).T,np.array([ranges]).T)
    c=np.cov(y.T)
    idxs=reindex(c)
    kmeans = KMeans(n_clusters=6, random_state=0, n_init="auto").fit(y)
    
    colors=kmeans.labels_.astype(float)
    global x
    x=np.average(y,axis=0)
    y=y[::100,:]
    
    colors=colors[::100]
    color=np.linalg.norm(y,axis=1)
    dims=ceil(sqrt(out_size/2+1))
    fig,axs=plt.subplots(dims,dims)
    if dims==1:
        axs=[axs]
    else:
        axs=axs.flatten()
    for i in range(out_size//2):
        j,k=idxs[i*2],idxs[i*2+1]
        axs[i].scatter(y[:,j],y[:,k],c=colors,s=20,cmap="plasma")
        #axs[i].set_xlim(0,1)
        #axs[i].set_ylim(0,1)
    
    
    
    fig.canvas.mpl_connect("motion_notify_event", lambda event: onclick(event,axs,model,mins,ranges,axs[-1],conn,idxs))
    fig.canvas.mpl_connect("button_press_event", set_var)
    fig.canvas.mpl_connect("button_release_event", clear_var)
    plt.pause(0)
if __name__ =="__main__":
    x=[]
    handle=None
    clicked=False
    env = gym.make("HandManipulateBlockRotateZDense-v1", render_mode="rgb_array")
    env.reset()
    rec,snd=Pipe()

    #fname="ae_trained_48_100_24_6.pkl"
    fname=sys.argv[1]
    
    Process(target=view,args=(fname,rec,)).start()
    Process(target=get_view,args=(snd,)).start()
    
    #view()