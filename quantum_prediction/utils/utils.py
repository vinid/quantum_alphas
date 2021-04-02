import pickle
from numpy.random import multivariate_normal
from numpy import hstack,array

def load_dataset(path):
    with open(path, "rb") as filino:
        data_dict = pickle.load(filino)
    return data_dict

def jumps(state1,state2,cov):
    errors = hstack((multivariate_normal(state1,cov,1)/4., multivariate_normal(state2,cov,1)/4.))[0]
    noised = hstack((state1,state2)) + errors
     
    norm1 = array([np.sum(noised[:4])]*4)
    norm2 = array([np.sum(noised[4:])]*4)
    final = noised/hstack( (norm1,norm2))
    return final


