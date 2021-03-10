from quantum_prediction.utils.utils import load_dataset
from quantum_prediction.utils.models import *
from keras.utils.np_utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import *
import logging
import gc
import numpy as np
import configparser
from sklearn.utils import shuffle
logging.getLogger().setLevel(logging.INFO)
config = configparser.ConfigParser()
config.read('config.ini')


NUM_OF_EXPERIMENTAL_RUNS = int(config['TRAINING']['num_of_experimental_runs'])

TRAINING_DATA_PATH = config['DATA']['training_path']
TESTING_DATA_PATH = config['DATA']['testing_path']
VALIDATION_DATA_PATH = config['DATA']['validation_path']

BATCH_SIZE = int(config['TRAINING']['batch_size'])
PATIENCE = int(config['TRAINING']['patience'])
MONITOR = config['TRAINING']['monitor']
VALIDATION_SPLIT = float(config['TRAINING']['validation_split'])
EPOCHS = int(config['TRAINING']['epochs'])
NUM_CLASSES = int(config['TRAINING']['num_classes'])

logging.info("Loading Training Set")
training_dict = load_dataset(TRAINING_DATA_PATH)
logging.info("Loading Test Set")
testing_dict = load_dataset(TESTING_DATA_PATH)
logging.info("Loading Valid Set")
validation_dict = load_dataset(VALIDATION_DATA_PATH)

logging.info("Data was loaded")

# we need to transform the continuous labels into one-hot-encoded
label_encoder = LabelEncoder()
training_labels = label_encoder.fit_transform(training_dict["labels"])
categorical_labels = to_categorical(training_labels)

# validation labels
validation_labels = label_encoder.transform(validation_dict["labels"])
categorical_validation_labels = to_categorical(validation_labels)

# testing labels
testing_labels = label_encoder.transform(testing_dict["labels"])
categorical_testing_labels = to_categorical(testing_labels)

# standard early stopping
es = EarlyStopping(monitor=MONITOR, patience=PATIENCE, mode='min', verbose=0, restore_best_weights=True)


train_i1, train_i2, train_i3, train_i4, train_labels = shuffle(training_dict["s1"],
               training_dict["s2"],
               training_dict["l1"],
               training_dict["l2"], categorical_labels)

train_input = np.column_stack([train_i1, train_i2, train_i3, train_i4])
del training_dict
valid_i1, valid_i2, valid_i3, valid_i4, valid_labels = shuffle(validation_dict["s1"],
               validation_dict["s2"],
               validation_dict["l1"],
               validation_dict["l2"], categorical_validation_labels)

valid_input = np.column_stack([valid_i1, valid_i2, valid_i3, valid_i4])
del validation_dict

test_i1, test_i2, test_i3, test_i4, test_labels = (testing_dict["s1"],
               testing_dict["s2"],
               testing_dict["l1"],
               testing_dict["l2"], categorical_testing_labels)

testing_input = np.column_stack([test_i1, test_i2, test_i3, test_i4])
del testing_dict

collect_accuracies = []
collect_f1_scores = []
for i in range(0, NUM_OF_EXPERIMENTAL_RUNS):
    print("We are training model " + str(i) + " of " + str(NUM_OF_EXPERIMENTAL_RUNS))

    # instantiate the model
    model = baseline_network(NUM_CLASSES)

    model.fit(train_input, train_labels,
              batch_size=BATCH_SIZE,
              epochs=EPOCHS,
              validation_data=(valid_input, valid_labels),
              callbacks=[es],
              verbose=1,
              shuffle=True)

    predictions = model.predict(testing_input, batch_size=BATCH_SIZE, verbose=1)
    predicted_classes = predictions.argmax(axis=-1)
    testing_labels = np.argmax(test_labels, axis=1)

    collect_accuracies.append(accuracy_score(testing_labels, predicted_classes))
    collect_f1_scores.append(f1_score(testing_labels, predicted_classes, average="macro"))

    logging.info("Current Accuracy: " + str(np.average(collect_accuracies)))
    logging.info("Current F1 Score: " + str(np.average(collect_f1_scores)))

    del model
    gc.collect()

print("Experiment Ended. Average Accuracy: " + str(np.average(collect_accuracies)))
print("Experiment Ended. Average F1: " + str(np.average(collect_f1_scores)))
print("Files:")
print(TRAINING_DATA_PATH)
print(VALIDATION_DATA_PATH)
print(TESTING_DATA_PATH)