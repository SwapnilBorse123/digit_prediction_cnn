#importing the required libraries
import numpy as np
# from mnist import MNIST
from keras.utils import np_utils
from keras.datasets import mnist

# to avoid the randomness
np.random.seed(5)

#laoding the MNIST dataset into training and testing variables
#mndata = MNIST('./')
#images_train, labels_train = mndata.load_training()
#images_test, labels_test = mndata.load_testing()

# Getting data as a ndarray 
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Converting data into 60000 tensors of 28*28*1 shape
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')

# Converting the pixel values in the range of 0 to 1
X_train = X_train / 255
X_test = X_test / 255

# Since the labels are categorical 0 - 9 we will one hot encode it using utils
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

num_classes = y_test.shape[1]

# Convolutional Neural Network
# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Step 1 - Convolution
# * Applying 32 filters[feature detectors] of size 5*5
# * Using the ReLU function for activation of neurons[to remove the negative values and have non-linearity]
classifier.add(Convolution2D(32, 5, 5, input_shape = (28, 28, 1), activation = 'relu'))

# Step 2 - Pooling
# * Adding max pooling to reduce the size of feature maps and size of flattened layer
# With max pooling we reduce complexity make it less computation intensive and preserve spatial features
# Generally, as a rule of thumb we take a 2*2 window to avoid loss of features
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
# * Using the Dense function to make a fully connected layer
# * Using a thumb rule of selecting number of neurons between i/p and o/p nodes. [Trial and Error]
classifier.add(Dense(output_dim = 64, activation = 'relu'))

# * Using the softmax function since we have only 10 categories
classifier.add(Dense(output_dim = 10, activation = 'softmax'))

# Compiling the CNN
# * Using the stochastic gradient descent optimizer - adam
# * Using cross entropy as cost function. Binary because we have only two categories. 
# categorical_crossentropy if we have more than two outcomes. Using cross-entropy due to it's logarithmic
# component
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# Training the model on the preprocessed data
classifier.fit(X_train, y_train, validation_data=(X_test, y_test), epochs = 5, batch_size = 200)

# Evaluating the trained model
classifier.evaluate(X_test, y_test)