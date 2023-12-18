'''scp -J cookjos@access.engr.oregonstate.edu cookjos@graf200-17.engr.oregonstate.edu:dexterity/data/c* data/'''

import pickle as pkl
import numpy as np

for i in range(8):
    idx=i

    with open("data/w"+str(idx)+".pkl","rb") as f:
        data=pkl.load(f)

        x,g=data
        print(len(g))
        a=np.argmax(g)
        print(x[a])
        print(max(g),min(g))