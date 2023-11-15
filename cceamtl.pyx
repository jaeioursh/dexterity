import numpy as np
import random
cimport cython

cdef extern from "math.h":
    double tanh(double m)

@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mul(double[:, :] mat, double[:] vec, double[:] out):
    cdef int colIndex, rowIndex
    cdef double sum = 0
    for rowIndex in range(mat.shape[0]):
        sum = 0
        for colIndex in range(mat.shape[1]):
            sum += mat[rowIndex, colIndex] * vec[colIndex]
        out[rowIndex] = sum
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef addInPlace(double[:] vec, double[:] other):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] += other[index]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef addInPlaceMat(double[:,:] mat, double[:,:] other):
    cdef int colIndex, rowIndex
    for rowIndex in range(mat.shape[0]):
        for colIndex in range(mat.shape[1]):
            mat[rowIndex, colIndex] += other[rowIndex, colIndex]
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef tanhInPlace(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = tanh(vec[index])
        
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef reluInPlace(double[:] vec):
    cdef int index
    for index in range(vec.shape[0]):
        vec[index] = vec[index] * (vec[index] > 0)
     
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mutate(double[:] vec, double m, double mr):
    shape = [vec.shape[0]]
    npMutation = np.random.standard_cauchy(shape)
    npMutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:] mutation = npMutation
    addInPlace(vec, mutation)
    
@cython.boundscheck(False)  # Deactivate bounds checking
@cython.wraparound(False)   # Deactivate negative indexing. 
cdef mutateMat(double[:,:] mat, double m, double mr):
    shape = [mat.shape[0], mat.shape[1]]
    npMutation = m * np.random.standard_cauchy(shape)
    npMutation *= np.random.uniform(0, 1, shape) < mr
    cdef double[:,:] mutation = npMutation
    addInPlaceMat(mat, mutation)

@cython.auto_pickle(True)        
cdef class Evo_MLP:
    cdef public double[:,:] inToHiddenMat
    cdef public double[:] inToHiddenBias
    cdef public double[:,:] hiddenToOutMat
    cdef public double[:] hiddenToOutBias
    cdef public double[:] hidden
    cdef public double[:] out
    cdef public object npInToHiddenMat
    cdef public object npInToHiddenBias
    cdef public object npHiddenToOutMat
    cdef public object npHiddenToOutBias
    cdef public object npHidden
    cdef public object npOut
    cdef public int input_shape
    cdef public int num_outputs
    cdef public int num_units
    cdef public double fitness
    cdef public list life

    cdef public list D
    cdef public list G
    cdef public list Z
    cdef public list S 
    cdef public int used
    
    def __init__(self, input_shape, num_outputs, num_units=16):
        self.input_shape = input_shape
        self.num_outputs = num_outputs
        self.num_units = num_units
        self.fitness = 0.0
        self.life = []
        self.D=[]
        self.G=[]
        self.Z=[]
        self.S=[]
        self.used=0

        # XAVIER INITIALIZATION
        stdev = (3/ input_shape) ** 0.5
        self.npInToHiddenMat = np.random.uniform(-stdev, stdev, (num_units, input_shape))
        self.npInToHiddenBias = np.random.uniform(-stdev, stdev, num_units)
        stdev = (3/ num_units) ** 0.5
        self.npHiddenToOutMat = np.random.uniform(-stdev, stdev, (num_outputs, num_units))
        self.npHiddenToOutBias = np.random.uniform(-stdev, stdev, num_outputs)
        
        self.npHidden = np.zeros(num_units)
        self.npOut = np.zeros(num_outputs)
        
        self.inToHiddenMat = self.npInToHiddenMat
        self.inToHiddenBias = self.npInToHiddenBias
        self.hiddenToOutMat = self.npHiddenToOutMat
        self.hiddenToOutBias = self.npHiddenToOutBias
        self.hidden = self.npHidden
        self.out = self.npOut
    def __getstate__(self):
        return (self.inToHiddenMat.base,
        self.inToHiddenBias.base,
        self.hiddenToOutMat.base,
        self.hiddenToOutBias.base,
        self.hidden.base,
        self.out.base,
        self.input_shape,
        self.num_outputs,
        self.num_units
        )
    def __setstate__(self,x):
        self.inToHiddenMat ,self.inToHiddenBias,self.hiddenToOutMat,self.hiddenToOutBias, self.hidden,self.out,self.input_shape,self.num_outputs,self.num_units = x
    cpdef get_action(self, double[:] state):
        mul(self.inToHiddenMat, state, self.hidden)
        addInPlace(self.hidden, self.inToHiddenBias)
        reluInPlace(self.hidden)
        mul(self.hiddenToOutMat, self.hidden, self.out)
        addInPlace(self.out, self.hiddenToOutBias)
        tanhInPlace(self.out)
        return self.out #change

    cpdef mutate(self):
        cdef double m = 1.0
        cdef double mr = 0.05
        mutateMat(self.inToHiddenMat, m, mr)
        mutate(self.inToHiddenBias, m, mr)
        mutateMat(self.hiddenToOutMat, m, mr)
        mutate(self.hiddenToOutBias, m, mr)

        
    cpdef copyFrom(self, other):
        self.input_shape = other.input_shape
        self.num_outputs = other.num_outputs
        self.num_units = other.num_units 
        
        cdef double[:,:] newInToHiddenMat = other.npInToHiddenMat
        self.inToHiddenMat[:] = newInToHiddenMat
        cdef double[:] newInToHiddenBias = other.npInToHiddenBias
        self.inToHiddenBias[:] = newInToHiddenBias
        cdef double[:,:] newHiddenToOutMat = other.npHiddenToOutMat
        self.hiddenToOutMat[:] = newHiddenToOutMat
        cdef double[:] newHiddenToOutBias = other.npHiddenToOutBias
        self.hiddenToOutBias[:] = newHiddenToOutBias
        

        
        

def initCcea(input_shape, num_outputs, num_units=16,num_types=10):
    def initCceaGo(data):
        data['Number of Types']=num_types
        number_agents = data['Number of Types']
        policyCount = data['Number of Policies']
        populationCol = [[Evo_MLP(input_shape,num_outputs,num_units) for i in range(policyCount)] for j in range(number_agents)] 
        data['Agent Populations'] = populationCol
    return initCceaGo
    
def clearFitness(data):
    populationCol = data['Agent Populations']
    number_agents = data['Number of Types']
    
    for agentIndex in range(number_agents):
        for policy in populationCol[agentIndex]:
            policy.fitness = 0
    
def assignCceaPolicies(data):
    number_agents = data['Number of Types']
    populationCol = data['Agent Populations']
    worldIndex = data["World Index"]
    policyCol = [None] * number_agents
    for idx in range(number_agents):
        
        policyCol[idx] = populationCol[idx][worldIndex]
    data["Agent Policies"] = policyCol
    

def assignBestCceaPolicies(data):
    number_agents = data['Number of Types']
    populationCol = data['Agent Populations']
    policyCol = [None] * number_agents
    for idx in range(number_agents):
        policyCol[idx] = max(populationCol[idx], key = lambda policy: policy.fitness)
        #policyCol[agentIndex] = populationCol[agentIndex][0]
    data["Agent Policies"] = policyCol




def rewardCceaPolicies(data):
    policyCol = data["Agent Policies"]
    number_agents = data['Number of Types']
    rewardCol = data["Agent Rewards"]
    for agentIndex in range(number_agents):
        policyCol[agentIndex].fitness = rewardCol[agentIndex]



    
cpdef evolveCceaPolicies(data): 
    cdef int number_agents = data['Number of Types']
    populationCol = data['Agent Populations']
    cdef int agentIndex, matchIndex, halfPopLen
    halfPopLen = int(len(populationCol[0])//2)
    for agentIndex in range(number_agents):

        population = populationCol[agentIndex]
        if 1:
            # Binary Tournament, replace loser with copy of winner, then mutate copy
            for matchIndex in range(halfPopLen):
                
                if population[2 * matchIndex].fitness > population[2 * matchIndex + 1].fitness:
                    population[2 * matchIndex + 1].copyFrom(population[2 * matchIndex])
                else:
                    population[2 * matchIndex].copyFrom(population[2 * matchIndex + 1])

                population[2 * matchIndex + 1].mutate()
                population[2 * matchIndex + 1].life=[]
                population[2 * matchIndex + 1].D=[] #new
                population[2 * matchIndex + 1].Z=[] #new
                population[2 * matchIndex + 1].S=[]
                population[2 * matchIndex + 1].fitness=0.0

                population[2 * matchIndex].used=0 #new
                population[2 * matchIndex + 1].used=0 #new
            
            random.shuffle(population)
            data['Agent Populations'][agentIndex] = population
            
    
            