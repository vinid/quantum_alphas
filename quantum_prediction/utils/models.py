import keras
from quantum_prediction.utils.support_layers import *


def baseline_regression(num_classes):
    input1 = keras.layers.Input(shape=(10,))
    output = keras.layers.Dense(num_classes, activation="softmax")(input1)
    model = keras.Model(input1, output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def baseline_regression_no_lambda(num_classes):
    input1 = keras.layers.Input(shape=(8,))
    output = keras.layers.Dense(num_classes, activation="softmax")(input1)
    model = keras.Model(input1, output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def baseline_network(num_classes):
    input1 = keras.layers.Input(shape=(10,))
    m1 = keras.layers.Dense(5, activation="relu")(input1)
    output = keras.layers.Dense(num_classes, activation="softmax")(m1)
    model = keras.Model(input1, output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def baseline_network_no_lambda(num_classes):
    input1 = keras.layers.Input(shape=(8,))
    m1 = keras.layers.Dense(5, activation="relu")(input1)
    output = keras.layers.Dense(num_classes, activation="softmax")(m1)
    model = keras.Model(input1, output)
    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def transformer_model(num_classes):
    input1 = keras.layers.Input(shape=(4,))
    input2 = keras.layers.Input(shape=(4,))
    input3 = keras.layers.Input(shape=(1,))
    input4 = keras.layers.Input(shape=(1,))

    m1 = keras.layers.Dense(4)(input1)
    m2 = keras.layers.Dense(4)(input2)
    m3 = keras.layers.Dense(4)(input3)
    m4 = keras.layers.Dense(4)(input4)

    stacked = tf.keras.backend.stack((m1, m2, m3, m4), axis=1)

    transformer_block = TransformerBlock(4, 4, 4)
    #transformer_block_2 = TransformerBlock(8, 4, 8)
    #transformer_block_3 = TransformerBlock(8, 4, 8)
    x = transformer_block(stacked)
    #x = transformer_block_2(x)
    #x = transformer_block_3(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model([input1, input2, input3, input4], output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def transformer_model_no_lambda(num_classes):
    input1 = keras.layers.Input(shape=(4,))
    input2 = keras.layers.Input(shape=(4,))

    input3 = keras.layers.Input(shape=(1,))
    input4 = keras.layers.Input(shape=(1,))

    m1 = keras.layers.Dense(4)(input1)
    m2 = keras.layers.Dense(4)(input2)


    stacked = tf.keras.backend.stack((m1, m2), axis=1)

    transformer_block = TransformerBlock(4, 2, 4)
    #transformer_block_2 = TransformerBlock(8, 4, 8)
    #transformer_block_3 = TransformerBlock(8, 4, 8)
    x = transformer_block(stacked)
    #x = transformer_block_2(x)
    #x = transformer_block_3(x)
    x = keras.layers.GlobalAveragePooling1D()(x)
    x = keras.layers.Dropout(0.1)(x)
    output = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model([input1, input2, input3, input4], output)

    model.compile(
        optimizer=keras.optimizers.RMSprop(),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model