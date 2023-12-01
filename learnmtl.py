import re
import numpy as np
#import tensorflow as tf
import numpy as np

from copy import deepcopy as copy
from logger import logger
import pyximport
pyximport.install()
from cceamtl import *
from itertools import combinations
#from math import comb
from collections import deque
from random import sample
from multiprocessing import Process,Queue

from hand import split_observation, setup_joints
JOINTS = setup_joints()

import torch
device = torch.device("cpu") 
torch.autograd.set_detect_anomaly(False)
torch.autograd.profiler.profile(False)
torch.autograd.profiler.emit_nvtx(False)
torch.set_num_threads(1)
print("threads: ",torch.get_num_threads())

import operator as op
from functools import reduce

def comb(n, r):
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


class Net:
    def __init__(self,hidden,lr,loss_fn,opti):
        learning_rate=lr
        self.model = torch.nn.Sequential(
            torch.nn.Linear(62, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1)
        )
        self.sig=torch.nn.Sigmoid()

        if loss_fn==0 or loss_fn==3:
            self.loss_fn = torch.nn.MSELoss(reduction='sum')
        elif loss_fn==1:
            self.loss_fn = self.alignment_loss
        elif loss_fn ==2:
            self.loss_fn = lambda x,y: self.alignment_loss(x,y) + torch.nn.MSELoss(reduction='sum')(x,y)


        if opti:
            self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    def feed(self,x):
        x=torch.from_numpy(x.astype(np.float32))
        pred=self.model(x)
        return pred.detach().numpy()
        
    
    def train(self,x,y,n=5,verb=0):
        x=torch.from_numpy(x.astype(np.float32))
        y=torch.from_numpy(y.astype(np.float32))
        pred=self.model(x)
        loss=self.loss_fn(pred,y)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.detach().item()

    def alignment_loss(self,o, t):

        ot=torch.transpose(o,0,1)
        tt=torch.transpose(t,0,1)

        O=o-ot
        T=t-tt

        align = torch.mul(O,T)
        #print(align)
        align = self.sig(align)
        loss = -torch.mean(align)
        return loss
    
def robust_sample(data,n):
        if len(data)<n: 
            smpl=data
        else:
            smpl=sample(data,n)
        return smpl

class learner:
                    #total agents, subteam size
    def __init__(self,nagents,train_flag,params):
        #params=[0,0,0,0,0]+params
        params+=[53, 0.40, 0.45]
        self.lr, self.hidden, self.batch, self.replay_size,opti,polh,m,mr= params
        self.hidden,self.batch,self.replay_size,opti,polh=[int (q) for q in [self.hidden,self.batch,self.replay_size,opti,polh]]

        self.idx=0

        self.train_flag=train_flag
        self.log=logger()
        self.nagents=nagents
        self.hist=[deque(maxlen=self.replay_size) for i in range(nagents)]
        self.zero=[deque(maxlen=self.replay_size) for i in range(nagents)]
        self.itr=0
        self.team=[]
        self.index=[]
        self.Dapprox=[Net(self.hidden,self.lr,train_flag-1,opti) for i in range(self.nagents)]

        self.every_team=[[i for i in range(nagents)]]
        self.test_teams=self.every_team
        self.team=self.every_team
        
        self.data={}
        self.pop_size=32
        self.data["Number of Policies"]=self.pop_size #pop size
        self.data["World Index"] = 0
        
        #policy shape
        if train_flag>=0:
            initCcea(input_shape=62, num_outputs=1,num_types=20, num_units=polh,m=m,mr=mr)(self.data)
        else:    
            initCcea(input_shape=62, num_outputs=20,num_types=1, num_units=polh,m=m,mr=mr)(self.data)
        

    def act(self,S,data):
        policyCol=data["Agent Policies"]
        A=[]

        #if self.train_flag>=0:
        #    states = split_observation(JOINTS, S)  #["observation"])
        #else:
        states = S

        for s,pol in zip(states,policyCol):
            
            a = pol.get_action(s)
            a=np.asarray(a)[0]
            
            A.append(a)
        #print(A)
        return np.array(A)
    

    def randomize(self):
        length=len(self.every_team)
        teams=[]
        
        idx=np.random.choice(length)
        t=self.every_team[idx].copy()
        #np.random.shuffle(t)
        teams.append(t)
        self.team=teams
        #self.team=np.random.randint(0,self.types,self.nagents)
    
            


    def set_teams(self,N,rand=0):
        if N >= len(self.every_team):
            self.team=self.every_team
            return
        if len(self.team)==0:
            self.index = np.random.choice(len(self.every_team), N, replace=False)  
        elif 0:
            self.index=self.minmax()
        else:
            i=np.random.randint(0,len(self.every_team))
            while i in self.index:
                i=np.random.randint(0,len(self.every_team))
            if rand:
                j=np.random.randint(0,len(self.index))
            elif 1:
                j=self.minmaxsingle()
            else:
                j=self.most_similar()
            self.index[j]=i
        
            


        self.index=np.sort(self.index)
        self.team=[self.every_team[i] for i in self.index]


    def save(self,fname="log.pkl"):
        print("saved")
        self.log.save(fname)
        #print(self.Dapprox[0].model.state_dict()['4.bias'].is_cuda)
        #netinfo={i:self.Dapprox[i].model.state_dict() for i in range(len(self.Dapprox))}
        #torch.save(netinfo,fname+".mdl")

  
    def proc(self,q,idx,env):

        self.data["World Index"]=idx
            
        #for agent_idx in range(self.types):
        data=[]
        
        s = env.reset() 
        s=s[0]
        done=False 
        #assignCceaPoliciesHOF(env.data)
        assignCceaPolicies(self.data)
        S,A=[],[]
        R=0.0
        for i in range(3):
            self.itr+=1
            if self.train_flag>=0:
                #agent_states = split_observation(JOINTS, s)
                agent_states = np.array([s["observation"]]*20)
            else:
                agent_states = np.array([s["observation"]])
            action=self.act(agent_states,self.data)
            action = np.array(action).flatten()

            S.append(agent_states)
            A.append(action)
            s, r, terminated, truncated, info = env.step(action)
            R+=r
            done = terminated or truncated
            #S,A=[S[-1]],[A[-1]]
        data.append([r,S,A])
            
            #G.append(g)
        if q is not None:
            q.put([data,idx])
        return [data,idx]
    
    def view(self,env):
        frames=[]
        R=[]
        for idx in range(self.pop_size):
            self.data["World Index"]=idx
                
            
            s = env.reset() 
            s=s[0]
            assignCceaPolicies(self.data)
            S,A=[],[]
            for i in range(100):
                if self.train_flag>=0:
                    agent_states = np.array([s["observation"]]*20)

                else:
                    agent_states = np.array([s["observation"]])
                action=self.act(agent_states,self.data,0)
                action = np.array(action).flatten()

                s, r, terminated, truncated, info = env.step(action)
                frames.append(env.render())
            R.append(r)
            print(r)
        return frames, R
    
    #train_flag=-1 - Single Agent        
    #train_flag=0 - G                        
    #train_flag=1 - gapprox
    #train_flag=2 - align
    #train_flag=3 - g+align
    #train_flag=4 - fitness critic
    
    def run(self,env,parallel=True):
        self.idx+=1
        train_flag=self.train_flag
        populationSize=len(self.data['Agent Populations'][0])
        pop=self.data['Agent Populations']
        #team=self.team[0]
        G=[]
        if parallel:
            procs=[]
            team_data=[]
            q = Queue()
            for idx,e in zip(range(populationSize),env):
                p = Process(target=self.proc, args=(q, idx, e))
                procs.append(p)
                p.start()
            for p in procs:
                ret = q.get() # will block
                team_data.append(ret)
            for p in procs:
                p.join()
        else:
            team_data=map(self.proc,[None]*self.pop_size,range(self.pop_size),[env]*self.pop_size)
        pols=self.data["Agent Populations"] 
        G=[]
        for data in team_data: # runs from each pop
            
            data,idx=data
            
                
            for g,S,A in data: #for each sample
                G.append(g)
                for i in range(len(S[0])): #for each agent
                        #d=r[i]
                        
                    pols[i][idx].G.append(g)
                    
                    #pols[i].D.append(g)
                    #pols[i][idx].S.append([])
                    for j in range(len(S)): #for each time step
                        sa=np.append(S[j][i],A[j][i])
                        z=[sa,g]
                        #if d!=0:
                        if train_flag==4:
                            self.hist[i].append(z)
                        #else:
                        #    self.zero[team[i]].append(z)
                        pols[i][idx].S.append(sa)
                    if train_flag!=4:
                        self.hist[i].append(z)
                    pols[i][idx].Z=[sa]
        if train_flag>0:
            self.updateD()  # env)

        for t in range(self.nagents):#np.unique(np.array(self.team)):
            #if train_flag==1:
            #    S_sample=self.state_sample(t)

            for p in pop[t]:
                
                if self.idx%250==0:
                    p.m/=2.0
                    p.mr/=2.0
                    
                if  train_flag==0:
                    p.G=[np.sum(p.G)]
                if  train_flag==1 or train_flag==2 or train_flag==3:
                    p.G=[self.Dapprox[t].feed(np.array(p.Z[0]))]
                if train_flag==4:
                    p.G=[np.max(self.Dapprox[t].feed(np.array(p.S)))]
                    #p.D=[(self.Dapprox[t].feed(np.array(p.S[i])))[-1] for i in range(len(p.S))]
                    #print(p.D)
                p.fitness=np.sum(p.G)
                    
                p.D=[]
                p.S=[]
                p.Z=[]
                p.G=[]
        evolveCceaPolicies(self.data)

        #self.log.store("reward",max(G))      
        return max(G)

    

    def updateD(self):
        
        for i in np.unique(np.array(self.team)):
            for q in range(100): #num batches
                S,A,D=[],[],[]
                SAD=robust_sample(self.hist[i],self.batch) #batch size
                
                for samp in SAD:
                    S.append(samp[0])
                    D.append([samp[1]])
                S,D=np.array(S),np.array(D)
                Z=S#np.hstack((S,A))
                self.Dapprox[i].train(Z,D)


    def approx(self,p,t,S):
        
        A=[p.get_action(s) for s in S]
        A=np.array(A)
        Z=np.hstack((S,A))
        D=self.Dapprox[t].feed(Z)
        fit=np.sum(D)
        #print(fit)
        p.fitness=fit

    def put(self,key,data):
        self.log.store(key,data)


   
    





    
