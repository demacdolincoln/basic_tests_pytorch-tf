import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import SparseCategoricalCrossentropy

from load_csv import split_data

logdir = "./runs"

x_train, x_test, y_train, y_test = split_data("UCI_Credit_Card.csv.zip")

data_train = tf.data.Dataset.from_tensor_slices((x_train.reshape((x_train.shape[0], 1, x_train.shape[1])), y_train)).shuffle(128).batch(3000)
data_test = tf.data.Dataset.from_tensor_slices((x_test.reshape((x_test.shape[0], 1, x_test.shape[1])), y_test))

tb_callback = keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1)

model = keras.Sequential(layers=[
    Dense(128, activation="relu"),
    Dense(64, activation="relu"),
    Dense(2)
])

model.compile(
    optimizer=Adam(learning_rate=1e-6),
    loss = SparseCategoricalCrossentropy(),
    metrics=["accuracy"]
)

model.fit(
    data_train,
    epochs=10,
    validation_data=data_test,
    callbacks=[tb_callback]
    )
