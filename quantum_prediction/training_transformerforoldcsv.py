import pickle
from utils.models import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import logging
import numpy as np
import configparser
from sklearn.utils import shuffle
import pandas as pd

config = configparser.ConfigParser()
config.read('config.ini')

TRAINING_DATA_PATH = config['DATA']['training_path']
TESTING_DATA_PATH = config['DATA']['testing_path']
BATCH_SIZE = int(config['TRAINING']['batch_size'])
PATIENCE = int(config['TRAINING']['patience'])
MONITOR = config['TRAINING']['monitor']
VALIDATION_SPLIT = float(config['TRAINING']['validation_split'])
EPOCHS = int(config['TRAINING']['epochs'])


keras.backend.clear_session()

data_dict = pd.read_csv(TRAINING_DATA_PATH, nrows = 12000000)

testing_dict =  pd.read_csv(TESTING_DATA_PATH,nrows = 6000000)

logging.info("Data was loaded")

# we need to transform the continuous labels into one-hot-encoded
label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(data_dict["alphas"])
categorical_labels = to_categorical(integer_labels)

# testing labels
testing_labels = label_encoder.transform(testing_dict["alphas"])
categorical_testing_labels = to_categorical(testing_labels)

# standard early stopping
es = EarlyStopping(monitor=MONITOR, patience=PATIENCE, mode='min', verbose=1, restore_best_weights=True)

# instantiate the model
model = transformer_model()

i1, i2, i3, i4, labels = shuffle(data_dict[[ 'in_prob1', 'in_prob2', 'in_prob3', 'in_prob4']].values,
                                 data_dict[ ['fin_prob1','fin_prob2', 'fin_prob3', 'fin_prob4']].values,
                                 data_dict["L_in"].values,
                                 data_dict["L_fin"].values,
                                 categorical_labels)

train_input = [i1, i2, i3, i4]


model.fit(train_input, labels,
          batch_size=200,
          epochs=EPOCHS,
          validation_split=VALIDATION_SPLIT,
          callbacks=[es],
          shuffle=True)

testing_input = [data_dict[[ 'in_prob1', 'in_prob2', 'in_prob3', 'in_prob4']].values,
                                 data_dict[ ['fin_prob1','fin_prob2', 'fin_prob3', 'fin_prob4']].values,
                                 data_dict["L_in"].values,
                                 data_dict["L_fin"].values,
                                 categorical_labels]



print(model.evaluate(testing_input, categorical_testing_labels))