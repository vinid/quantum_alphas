import pickle
from numpy.random import multivariate_normal
import numpy as np

def load_dataset(path):
    with open(path, "rb") as filino:
        data_dict = pickle.load(filino)
    return data_dict


def noise_single_state(state1, loc=0, dev=0.2):
    noised = state1 + np.random.normal(loc, dev, 4)
    norm1 = np.sum(noised)
    final = noised / norm1
    return final


def noise_states(state1, state2, cov):
    errors = np.hstack((multivariate_normal(state1, cov,1)/4., multivariate_normal(state2, cov,1)/4.))[0]
    noised = np.hstack((state1, state2)) + errors
     
    norm1 = np.array([np.sum(noised[:4])]*4)
    norm2 = np.array([np.sum(noised[4:])]*4)
    final = noised/np.hstack((norm1, norm2))
    return final[0:4], final[4:]

def noise_states_list(s1, loc=0, dev=0.2):
    return np.array([noise_single_state(s, loc, dev) for s in s1])


