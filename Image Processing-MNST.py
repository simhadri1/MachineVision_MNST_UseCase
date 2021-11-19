import tensorflow as tf
print(tf.__version__)

#  loading MNST data set from tf.keras datasets api

mnist = tf.keras.datasets.fashion_mnist

# calling load data to two sets of two lists , ie training and testing

(training_images, training_labels), (test_images, test_labels) = mnist.load_data()

# see how the image looks like

import numpy as np
np.set_printoptions(linewidth=200)
import matplotlib.pyplot as plt
plt.imshow(training_images[0])
print(training_labels[0])
print(training_images[0])

#normalize the image beween zero and 1

training_images = training_images/255.0
test_images = test_images/255.0

# design of the model

model = tf.keras.models.Sequential([tf.keras.layers.Flatten(),
                                    tf.keras.layers.Dense(128, activation=tf.nn.relu),
                                    tf.keras.layers.Dense(10,activation=tf.nn.softmax)])

# build model by compiling with optimizer and loss function

model.compile(optimizer=tf.optimizers.Adam(),
                loss = 'sparse_categorical_crossentropy',
                metrics=['accuracy'])

model.fit(training_images, training_labels, epochs=5)
#check model accuracy
model.evaluate(test_images, test_labels)

# classifcation for each of the image

classifications = model.predict(test_images)

print(classifications[2])

print (test_labels[2])

len(test_images)+len(training_images)