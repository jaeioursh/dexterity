import pickle as pkl
import gzip

class logger:
    def __init__(self):
        self.data={}
    
    def store(self,tag,data,idx=None):
        
        if not tag in self.data:
            self.data[tag]=[]
        
        if idx is None:
            self.data[tag].append(data)
        elif idx<0:
            self.data[tag]=data
        else:
            if len(self.data[tag])<=idx:
                self.data[tag].append([])
            #print(idx,self.data[tag])
            self.data[tag][idx].append(data)
        

    def pull(self,tag):
        return self.data[tag]
    
    def clear(self,tag):
        self.data[tag]=[]

    def save(self,fname):
        with gzip.open(fname,"wb") as f:
            pkl.dump( self.data,f)
    def load(self,fname):
        with gzip.open(fname,"rb") as f:
            self.data=pkl.load(f)