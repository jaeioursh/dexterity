import numpy as np
import matplotlib.pyplot as plt
import matplotlib
#from math import comb
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42
plt.style.use('tableau-colorblind10')
#matplotlib.rcParams['text.usetex'] = True
import logger
DEBUG=1

#schedule = ["evo"+num,"base"+num,"EVO"+num]
#schedule = ["base"+num+"_"+str(q) for q in [0.0,0.25,0.5,0.75,1.0]]
AGENTS=5
ROBOTS=4
vals=sorted([0.8,1.0,0.6,0.3,0.2,0.1],reverse=True)
lbls={0:"D Rand.",1:"End State Aprx.",2:"Ctrfctl. Aprx.",3:"Fit Critic",4:"$D^\Sigma$",5:"$G^\Sigma$",-1:"Single Agent"}
if DEBUG:
    plt.subplot(1,2,1)
mint=1e9
colors={0:"G",1:"G apprx",2:"Align",3:"Both",4:"FC"}
lbls=colors
folder="data"
for q in [0]:#,1,2,3,4]:#,1]:
    T=[]
    print(q)
    for i in range(12):#range(8):
        log = logger.logger()
        
        try:
            log.load("data/Q"+str(q)+"_"+str(i)+".pkl")
        except:
            print("noload")
            print("data/Q"+str(q)+"_"+str(i)+".pkl")
            continue
    
        t=log.pull("reward")
        print(np.array(t).shape)

        t=np.array(t)
        mint=min(len(t),mint)
        if DEBUG and q==5:
            plt.subplot(1,2,1)
            plt.plot(t,label=str(i))
            print(i,q,t[-1])
        T.append(t)
    
    if DEBUG:
        leg=plt.legend()
        for legobj in leg.legendHandles:
            legobj.set_linewidth(4.0)
        plt.subplot(1,2,2)
        

    
    #R=np.mean(R,axis=0)
    T=[t[:mint] for t in T]
    BEST=np.max(T,axis=0)
    std=np.std(T,axis=0)/np.sqrt(120)
    T=np.mean(T,axis=0)
    X=[i*1 for i in range(len(T))]

    plt.plot(X,T,label=lbls[q])
    plt.fill_between(X,T-std,T+std,alpha=0.35, label='_nolegend_')

    #plt.ylim([0,1.15])
    plt.grid(True)


#plt.plot(X,[0.5]*101,"--")
#plt.plot(X,[0.8]*101,"--")
#plt.legend(["Random Teaming + Types","Unique Learners","Types Only","Max single POI reward","Max reward"])
plt.xlabel("Generation")

leg=plt.legend()
for legobj in leg.legendHandles:
    legobj.set_linewidth(4.0)
#leg=plt.legend(["Min","First Quartile","Median","Third Quartile","Max"])
#for legobj in leg.legendHandles:
#    legobj.set_linewidth(5.0)
plt.ylabel("$G$ ")

'''
if num[1]=="5":
    plt.title("5 agents, coupling req. of 2")
if num[1]=="8":
    plt.title("8 agents, coupling req. of 3")
'''
#plt.title("Team Performance Across \n Quartile Selection Methods")

plt.tight_layout()
#plt.savefig("figsv3/vis8_"+str(ROBOTS)+"_"+str(AGENTS)+".png")
plt.show()