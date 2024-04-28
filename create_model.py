from keras import saving
from keras.layers import Input, Conv2D, Lambda, Dense, Flatten,MaxPooling2D,Activation, Dropout, BatchNormalization, Layer
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
import cv2

saving.get_custom_objects().clear()

@saving.register_keras_serializable(package="MyLayers")
class L1(Layer):
    def __init__(self, **kwargs):
        super().__init__()
    #     self.w = self.add_weight(
    #         shape=(input_dim, units),
    #         initializer="random_normal",
    #         trainable=True,
    #     )
    #     self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.math.abs(inputs[0] - inputs[1])

def main():
    left_input = Input((62, 47, 3))
    right_input = Input((62, 47, 3))
    output_shape = 62

    # convnet = Sequential([
    #     Conv2D(1,5, input_shape=(62, 47, 3)),
    #     Activation('relu'),
    #     MaxPooling2D(),
    #     Flatten(),
    #     Dense(output_shape),
    #     Activation('sigmoid')
    # ])

    convnet = Sequential([
        Conv2D(30,5, input_shape=(62, 47, 3), padding="same"),
        Activation('relu'),
        Conv2D(30,3, padding="same"),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(30,3, padding="same"),
        Activation('relu'),
        MaxPooling2D(),
        Conv2D(40,2),
        Activation('relu'),
        Conv2D(40,2),
        Activation('relu'),
        Conv2D(40,2),
        Activation('relu'),
        MaxPooling2D(),
        Flatten(),
        Dense(50),
        BatchNormalization(),
        Dropout(0.5),
        Activation('relu'),
        Dense(output_shape),
        Activation('sigmoid')
    ])

    # Connect each 'leg' of the network to each input
    # Remember, they have the same weights
    encoded_l = convnet(left_input)
    # print(encoded_l.shape)
    encoded_r = convnet(right_input)
    # print(encoded_l.shape)

    L_layer = L1()
    L1_distance = L_layer([encoded_l, encoded_r])

    # Getting the L1 Distance between the 2 encodings
    # L1_layer = Lambda(custom_layer, output_shape=(3023, output_shape))

    # Add the distance function to the network
    # L1_distance = L1_layer([encoded_l, encoded_r])

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

    # Number of pairs per image
    pairs = 6
    # Let's create the new dataset to train on
    for i in range(len(label_list)):
        # for _ in range(pairs):
        #     compare_to = i
        #     while compare_to == i: # Make sure it's not comparing to itself
        #         compare_to = random.randint(0, X_train.shape[0] - 1)
        #     left_input.append(image_list[i])
        #     right_input.append(image_list[compare_to])
        #     if label_list[i] == label_list[compare_to]:# They are the same
        #         targets.append(1.)
        #     else:# Not the same
        #         targets.append(0.)
        # Half of the pairs should be the same person
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are the same person
            while compare_to == i or label_list[i] != label_list[compare_to]:
                compare_to = random.randint(0, X_train.shape[0] - 1)
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
            targets.append(1.)
        # Half of the pairs should be different people
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are diff ppl
            while compare_to == i or label_list[i] == label_list[compare_to]:
                compare_to = random.randint(0, X_train.shape[0] - 1)
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
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
    print("PERCENTAGE OF ONES IN targets", np.count_nonzero(targets)/targets.shape[0])

    # Test code
    test_image_list = np.split(X_test, X_test.shape[0])
    test_label_list = np.split(y_test, y_test.shape[0])

    test_left = []
    test_right = []
    test_targets = []

    # Let's create the new dataset to test on
    for i in range(len(test_label_list)):
        # for _ in range(pairs):
        #     compare_to = i
        #     while compare_to == i: # Make sure it's not comparing to itself
        #         compare_to = random.randint(0, X_test.shape[0] - 1)
        #     test_left.append(test_image_list[i])
        #     test_right.append(test_image_list[compare_to])
        #     if test_label_list[i] == test_label_list[compare_to]:# They are the same
        #         test_targets.append(1.)
        #     else:# Not the same
        #         test_targets.append(0.)
        # Half of the pairs should be the same person
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are the same person
            while compare_to == i or test_label_list[i] != test_label_list[compare_to]:
                compare_to = random.randint(0, X_test.shape[0] - 1)
            test_left.append(test_image_list[i])
            test_right.append(test_image_list[compare_to])
            test_targets.append(1.)
        # Half of the pairs should be different people
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are diff ppl
            while compare_to == i or test_label_list[i] == test_label_list[compare_to]:
                compare_to = random.randint(0, X_test.shape[0] - 1)
            test_left.append(test_image_list[i])
            test_right.append(test_image_list[compare_to])
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

    # Choose a few pairs of images from test set and visualize model performance
    print("Let's see how our model does on our favorite pair of test images")
    img = cv2.imread("datasets/lfw_home/lfw_funneled/Viara_Vike-Freiberga/Viara_Vike-Freiberga_0001.jpg")
    img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img2 = cv2.imread("datasets/lfw_home/lfw_funneled/Zahir_Shah/Zahir_Shah_0001.jpg")
    img2 = cv2.resize(img2, (47, 62), interpolation=cv2.INTER_NEAREST)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    fig = plt.figure(figsize=(10, 7))
    rows = 1
    cols = 2
    fig.add_subplot(rows, cols, 1)
    plt.imshow(img)
    plt.axis('off')
    plt.title("left image")
    fig.add_subplot(rows, cols, 2)
    plt.imshow(img2)
    plt.axis('off')
    plt.title("right image")
    plt.show()

    # print(test_targets)
    # print("label: ", test_targets[i])
    predicted_label = np.round((siamese_net.predict([np.array([img]), np.array([img2])]))[0])
    print("predicted label: ", predicted_label)

    # if test_targets[i] == predicted_label:
    #     print("YAYYYYYYY :))))) WOOOHOOOOO")
    # else:
    #     print("FAIL >:(((")
        

    # Save the model
    siamese_net.save('siamese_model.keras')

    # # serialize model to JSON
    # model_json = siamese_net.to_json()
    # with open("model.json", "w") as json_file:
    #     json_file.write(model_json)
    # # serialize weights to HDF5
    # siamese_net.save_weights("model.weights.h5")

if __name__ == "__main__":
    main()