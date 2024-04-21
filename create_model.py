from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation, Dropout
from keras.models import Model, Sequential
from keras.regularizers import l2
import tensorflow as tf
from keras.optimizers import Adam
# from skimage.io import imshow
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from grab_dataset import load_dataset

# def createSiameseNetwork():
    # # We have 2 inputs, 1 for each picture
    # left_input = Input((62, 47, 3))
    # right_input = Input((62, 47, 3))
    # output_shape = 18

    # # We will use 2 instances of 1 network for this task
    # convnet = Sequential([
    #     Conv2D(5,3, input_shape=(62, 47, 3)),
    #     Activation('relu'),
    #     MaxPooling2D(),
    #     Conv2D(5,3),
    #     Activation('relu'),
    #     MaxPooling2D(),
    #     Conv2D(7,2),
    #     Activation('relu'),
    #     MaxPooling2D(),
    #     Conv2D(7,2),
    #     Activation('relu'),
    #     Flatten(),
    #     Dense(output_shape),
    #     Activation('sigmoid')
    # ])
    # # Connect each 'leg' of the network to each input
    # # Remember, they have the same weights
    # encoded_l = convnet(left_input)
    # print(encoded_l.shape)
    # encoded_r = convnet(right_input)
    # print(encoded_l.shape)

    # # Getting the L1 Distance between the 2 encodings
    # L1_layer = Lambda(lambda tensor:K.abs(tensor[0] - tensor[1]))

    # # Add the distance function to the network
    # L1_distance = L1_layer([encoded_l, encoded_r])

    # prediction = Dense(1,activation='sigmoid')(L1_distance)
    # siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # optimizer = Adam(0.001, decay=2.5e-4)
    # #//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
    # siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])
    # return siamese_net
    # We have 2 inputs, 1 for each picture
left_input = Input((62, 47, 3))
right_input = Input((62, 47, 3))
output_shape = 62

# We will use 2 instances of 1 network for this task
convnet = Sequential([
    Conv2D(5,3, input_shape=(62, 47, 3)),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(5,3),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(7,2),
    Activation('relu'),
    MaxPooling2D(),
    Conv2D(7,2),
    Activation('relu'),
    Flatten(),
    Dense(output_shape),
    Activation('sigmoid')
])
# Connect each 'leg' of the network to each input
# Remember, they have the same weights
encoded_l = convnet(left_input)
# print(encoded_l.shape)
encoded_r = convnet(right_input)
# print(encoded_l.shape)

# Getting the L1 Distance between the 2 encodings
L1_layer = Lambda(lambda tensor:tf.math.abs(tensor[0] - tensor[1]), output_shape=(3023, output_shape))

# Add the distance function to the network
L1_distance = L1_layer([encoded_l, encoded_r])

prediction = Dense(1,activation='sigmoid')(L1_distance)
siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

optimizer = Adam(0.001, decay=2.5e-4)
#//TODO: get layerwise learning rates and momentum annealing scheme described in paperworking
siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

X, n_features, y, target_names, n_classes = load_dataset()

# stratify=target_names
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)


# First let's separate the dataset from 1 matrix to a list of matricies
image_list = np.split(X_train, X_train.shape[0])
label_list = np.split(y_train, y_train.shape[0])

left_input = []
right_input = []
targets = []

#Number of pairs per image
pairs = 5
#Let's create the new dataset to train on
for i in range(len(label_list)):
    for _ in range(pairs):
        compare_to = i
        while compare_to == i: #Make sure it's not comparing to itself
            compare_to = random.randint(0, X_train.shape[0] - 1)
        left_input.append(image_list[i])
        right_input.append(image_list[compare_to])
        if label_list[i] == label_list[compare_to]:# They are the same
            targets.append(1.)
        else:# Not the same
            targets.append(0.)
            
left_input = np.squeeze(np.array(left_input))
# left_input = np.array(left_input)
right_input = np.squeeze(np.array(right_input))
# right_input = np.array(right_input)
targets = np.squeeze(np.array(targets))
# targets = np.array(targets)

print("left_input: ", left_input.shape)
print("right_input: ", right_input.shape)
print("target: ", targets.shape)

# Test code
test_image_list = np.split(X_test, X_test.shape[0])
test_label_list = np.split(y_test, y_test.shape[0])

test_left = []
test_right = []
test_targets = []

#Let's create the new dataset to test on
for i in range(len(test_label_list)):
    for _ in range(pairs):
        compare_to = i
        while compare_to == i: #Make sure it's not comparing to itself
            compare_to = random.randint(0, X_test.shape[0] - 1)
        test_left.append(test_image_list[i])
        test_right.append(test_image_list[compare_to])
        if test_label_list[i] == test_label_list[compare_to]:# They are the same
            test_targets.append(1.)
        else:# Not the same
            test_targets.append(0.)

print("test_left: ", np.array(test_left).shape)
test_left = np.squeeze(np.array(test_left))
print("test_left: ", test_left.shape)

print("test_right ", np.array(test_right).shape)
test_right = np.squeeze(np.array(test_right))
print("test_right ", test_right.shape)

print("test_targets ", np.array(test_targets).shape)
test_targets = np.squeeze(np.array(test_targets))
print("test_targets ", test_targets.shape)

# siamese_net = createSiameseNetwork()
siamese_net.summary()
siamese_net.fit([left_input,right_input], targets,
          batch_size=16,
          epochs=30,
          verbose=1,
          validation_data=([test_left,test_right],test_targets))

print("Let's see how our model does on an arbitrary pair of test images")
plt.imshow(test_left[10])
plt.show()
plt.imshow(test_right[10])
plt.show()

print(test_targets)
print("label: ", test_targets[10])
predicted_label = siamese_net.predict([test_left,test_right])
print("predicted label: ", predicted_label[10])

if test_targets[10] == predicted_label:
    print("YAYYYYYYY :))))) WOOOHOOOOO")
else:
    print("FAIL >:(((")