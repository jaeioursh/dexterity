from test import test
from skopt import gp_minimize
from skopt import callbacks
import multiprocessing as mp
import pickle as pkl

test0=lambda x: test(20,3,1000,False,x)
testg=lambda x: test(20,0,1000,False,x)

def opt(test0,idx):
    #C=[(0.0001, 0.001),(3.0,120.0),(4.0,256.0),(100.0,100000.0),(0.7,1.3),
    C=[(10.0,128.0),(0.01,0.2),(0.01,0.2)]
    def saver(res):
        with open("data/c"+str(idx)+".pkl","wb") as f:
            data=[res.x_iters,res.func_vals]
            print("saving "+str(len(res.x_iters)))
            print(data)
            pkl.dump(data,f)
    res = gp_minimize(testg, C, n_calls=50,callback=[saver],n_jobs=3)#,acq_func="PI")
    print(res.x)
    print(res.fun)
    

procs=[]

for idx in range(3):
    p=mp.Process(target=opt,args=(test0,idx))
    p.start()
    procs.append(p)
for p in procs:
    p.join()