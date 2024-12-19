# =============================================================================
# PROBLEM C2
#
# Create a classifier for the MNIST Handwritten digit dataset.
# The test will expect it to classify 10 classes.
#
# Don't use lambda layers in your model.
#
# Desired accuracy AND validation_accuracy > 91%
# =============================================================================

import urllib.request
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow import keras
from tensorflow.keras.optimizers import Adam, RMSprop

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense


def solution_C2():
    mnist = tf.keras.datasets.mnist

    # YOUR CODE HERE
    (training_images, training_labels), (test_images,
                                         test_labels) = mnist.load_data()

    training_images = training_images / 255.0
    test_images = test_images / 255.0

    class myCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs={}):
            if(logs.get('accuracy') > 0.95 and logs.get('val_accuracy') > 0.95):
                print("\n Accuracy is on target more than 91%, stop training...")
                self.model.stop_training = True

    custom_callback = myCallback()

    # YOUR CODE HERE
    model = tf.keras.Sequential([
        tf.keras.layers.Flatten(input_shape=(28,28,1)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.fit(
        training_images,
        training_labels,
        epochs=20,
        validation_data=(test_images, test_labels),
        callbacks=[custom_callback])

    return model


# The code below is to save your model as a .h5 file.
# It will be saved automatically in your Submission folder.
if __name__ == '__main__':
    if __name__ == '__main__':
        model = solution_C2()
        model.save("model_C2.h5")
