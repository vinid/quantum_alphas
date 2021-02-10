import pickle
from quantum_prediction.utils.models import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
import configparser

config = configparser.ConfigParser()
config.read('config.ini')

TRAINING_DATA_PATH = config['DATA']['training_path']
TESTING_DATA_PATH = config['DATA']['testing_path']
BATCH_SIZE = int(config['TRAINING']['batch_size'])
PATIENCE = int(config['TRAINING']['patience'])
MONITOR = config['TRAINING']['monitor']
VALIDATION_SPLIT = float(config['TRAINING']['validation_split'])
EPOCHS = int(config['TRAINING']['epochs'])

with open(TRAINING_DATA_PATH, "rb") as filino:
    data_dict = pickle.load(filino)

logging.info("Data was loaded")

# we need to transform the continuous labels into one-hot-encoded
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(data_dict["labels"])
categorical_labels = to_categorical(integer_labels)

# standard early stopping
es = EarlyStopping(monitor=MONITOR, patience=PATIENCE, mode='min', verbose=1, restore_best_weights=True)

# instantiate the model
model = transformer_model()

train_input = [np.array(data_dict["s1"]),
               np.array(data_dict["s2"]),
               np.array(data_dict["l1"]),
               np.array(data_dict["l2"])]

model.fit(train_input, categorical_labels,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT,
          callbacks=[es], shuffle=True)



# train_input = np.column_stack([np.array(data_dict["s1"]),
#            np.array(data_dict["s2"]),
#            np.array(data_dict["l1"]),
#            np.array(data_dict["l2"])])