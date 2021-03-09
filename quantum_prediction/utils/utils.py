import pickle

def load_dataset(path):
    with open(path, "rb") as filino:
        data_dict = pickle.load(filino)
    return data_dict

