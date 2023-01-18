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
    def __init__(self,hidden=50):
        learning_rate=1e-3
        self.model = torch.nn.Sequential(
            torch.nn.Linear(22, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden,1)
        )
        self.loss_fn = torch.nn.MSELoss(reduction='sum')

        #self.optimizer = torch.optim.RMSprop(self.model.parameters(), lr=learning_rate)
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
    
def robust_sample(data,n):
        if len(data)<n: 
            smpl=data
        else:
            smpl=sample(data,n)
        return smpl

class learner:
                    #total agents, subteam size
    def __init__(self,nagents,types):
        self.log=logger()
        self.nagents=nagents
        self.hist=[deque(maxlen=20000) for i in range(types)]
        self.zero=[deque(maxlen=100) for i in range(types)]
        self.itr=0
        self.types=types
        self.team=[]
        self.index=[]
        self.Dapprox=[Net() for i in range(self.types)]

        self.every_team=self.many_teams()
        self.test_teams=self.every_team

        self.data={}
        
        self.data["Number of Policies"]=32 #pop size
        self.data["World Index"] = 0
        
        #policy shape
        initCcea(input_shape=22, num_outputs=1, num_units=20, num_types=types)(self.data)
        

    def act(self,S,data,trial):
        policyCol=data["Agent Policies"]
        A=[]

        try:
            iter(S)
            S = S[0]
        except:
            pass
        states = split_observation(JOINTS, S)  #["observation"])

        for s,pol in zip(states,policyCol):
  
            a = pol.get_action(s)*2.0
            A.append(a)
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

    #train_flag=0 - D
    #train_flag=1 - Neural Net Approx of D
    #train_flag=2 - counterfactual-aprx
    #train_flag=3 - fitness critic
    #train_flag=4 - D*
    #train_flag=5 - G*
    def run(self,env,train_flag):

        populationSize=len(self.data['Agent Populations'][0])
        pop=self.data['Agent Populations']
        #team=self.team[0]
        G=[]
        
        for worldIndex in range(populationSize):
            self.data["World Index"]=worldIndex
            
            #for agent_idx in range(self.types):
            for team in self.team:
                s = env.reset() 
                done=False 
                #assignCceaPoliciesHOF(env.data)
                assignCceaPolicies(self.data,team)
                S,A=[],[]
                while not done:
                    self.itr+=1
                    
                    action=self.act(s,self.data,0)
                    action = np.array(action).flatten()
                    agent_states = split_observation(JOINTS, s)
                    S.append(agent_states)
                    A.append(action)
                    s, r, terminated, truncated, info = env.step(action)
                    done = terminated or truncated
                #S,A=[S[-1]],[A[-1]]
                pols=self.data["Agent Policies"] 
                g=r#env.data["Global Reward"]
                for i in range(len(S[0])):

                    #d=r[i]
                    
                    pols[i].G.append(g)
                    
                    #pols[i].D.append(g)
                    pols[i].S.append([])
                    for j in range(len(S)):
                        z=[S[j][i],A[j][i],g]
                        #if d!=0:
                        self.hist[team[i]].append(z)
                        #else:
                        #    self.zero[team[i]].append(z)
                        pols[i].S[-1].append(S[j][i])
                    pols[i].Z.append(S[-1][i])
                        
                G.append(g)
        

        if train_flag==1 or train_flag==2 or train_flag==3:
            self.updateD()  # env)
        train_set=np.unique(np.array(self.team))
        for t in np.unique(np.array(self.team)):
            #if train_flag==1:
            #    S_sample=self.state_sample(t)

            for p in pop[t]:
                
                #d=p.D[-1]
                if train_flag==4:
                    p.fitness=np.sum(p.D)
                    p.D=[]
                if  train_flag==5:
                    p.fitness=np.sum(p.G)
                    p.G=[]
                if train_flag==3:
                    p.D=[np.max(self.Dapprox[t].feed(np.array(p.S[i]))) for i in range(len(p.S))]
                    #p.D=[(self.Dapprox[t].feed(np.array(p.S[i])))[-1] for i in range(len(p.S))]
                    #print(p.D)
                    p.fitness=np.sum(p.D)
                    p.S=[]
                if train_flag==1 or train_flag==2:
                    #self.approx(p,t,S_sample)
                    p.D=list(self.Dapprox[t].feed(np.array(p.Z)))
                    p.fitness=np.sum(p.D)
                    if train_flag==2:
                        p.fitness=np.sum(p.G)-np.sum(p.D)
                        p.G=[]
                    #print(p.fitness)
                    p.Z=[]
                    
                if train_flag==0:
                    d=p.D[-1]
                    p.fitness=d
        

        evolveCceaPolicies(self.data,train_set)

        #self.log.store("reward",max(G))      
        return max(G)

    

    def updateD(self):
        
        for i in np.unique(np.array(self.team)):
            for q in range(25): #num batches
                S,A,D=[],[],[]
                SAD=robust_sample(self.hist[i],100) #batch size
                
                for samp in SAD:
                    S.append(samp[0])
                    A.append(samp[1])
                    D.append([samp[2]])
                S,A,D=np.array(S),np.array(A),np.array(D)
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


   

            
    def many_teams(self):
        teams=[]
        C=comb(self.types,self.nagents)
        print("Combinations: "+str(C))
        if C<100:
            for t in combinations(range(self.types),self.nagents):
                teams.append(list(t))
        else:
            for i in range(100):
                teams.append(self.sample())

        return teams
    
    def sample(self):
        n,k=self.nagents,self.types
        return np.sort(np.random.choice(k,n,replace=False))




    