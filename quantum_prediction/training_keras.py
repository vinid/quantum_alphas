import pickle
from quantum_prediction.utils.models import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
import numpy as np

with open("../data/pure_training_data.pkl", "rb") as filino:
    data_dict = pickle.load(filino)

print("loaded")

label_encoder = LabelEncoder()
integer_labels = label_encoder.fit_transform(data_dict["labels"])
categorical_labels = to_categorical(integer_labels)
print("categorical")

es = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1, restore_best_weights=True)

model = transformer_model()

# train_input = np.column_stack([np.array(data_dict["s1"]),
#            np.array(data_dict["s2"]),
#            np.array(data_dict["l1"]),
#            np.array(data_dict["l2"])])

train_input = [np.array(data_dict["s1"]),
               np.array(data_dict["s2"]),
               np.array(data_dict["l1"]),
               np.array(data_dict["l2"])]

model.fit(train_input, categorical_labels, batch_size=300, epochs=10, validation_split=0.2, callbacks=[es], shuffle=True)

