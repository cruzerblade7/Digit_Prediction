#import libraries
import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Dropout, MaxPool2D, GlobalMaxPool2D
import matplotlib.pyplot as plt

#load in data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
K = len(set(y_train))
x_train, x_test = tf.expand_dims(x_train, -1), tf.expand_dims(x_test, -1)

#building the model
i = Input(shape = x_train[0].shape)
x = Conv2D(16, (3, 3), activation = 'relu', padding = 'same')(i)
x = MaxPool2D(2, 2)(x)
x = Conv2D(32, (3, 3), activation = 'relu', padding = 'same')(x)
x = GlobalMaxPool2D()(x)
x = Dense(128, activation = 'relu')(x)
x = Dense(K, activation = 'softmax')(x)

model = tf.keras.models.Model(i, x)

#training the model
model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)
r = model.fit(x_train, y_train, validation_data = (x_test, y_test), epochs = 12)

#plotting loss and accuracy
plt.subplot(1, 2, 1)
plt.plot(r.history["loss"], label = "loss")
plt.plot(r.history["val_loss"], label = "val_loss")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(r.history["accuracy"], label = "accuracy")
plt.plot(r.history["val_accuracy"], label = "val_accuracy")
plt.legend()
plt.show()

model.save('new_model')
