from keras import saving
from keras.layers import Input, Conv2D, Dense, Flatten, MaxPooling2D, Activation, Dropout, BatchNormalization, Layer
from keras.models import Model, Sequential
import tensorflow as tf
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.model_selection import train_test_split
from grab_dataset import load_dataset
import cv2
import time
import os
import uuid


saving.get_custom_objects().clear()
@saving.register_keras_serializable(package="MyLayers")
class L1(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    def call(self, inputs):
        return tf.math.abs(inputs[0] - inputs[1])


def predict_image(img_path, siamese_model, threshold=.8):
    score = 0

    # Read and preprocess input image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Read and preprocess anchor images to compare the input to
    # Anchor images are images of the same person from multiple angles
    anchor_path = "anchors"
    anchor_files = os.listdir(anchor_path)
    anchor_imgs = []
    for anchor in anchor_files:
        anchor_img = cv2.imread(f"{anchor_path}/{anchor}")
        anchor_img = cv2.resize(anchor_img, (47, 62), interpolation=cv2.INTER_NEAREST)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        anchor_imgs.append(anchor_img)

    # For each anchor image, use the model to predict whether the input matches
    for anchor in anchor_imgs:
        predicted_label = np.round(siamese_model.predict([np.array([img]), np.array([anchor])])[0])
        # if input matches anchor, predicted label = 1, else 0
        score += int(predicted_label)

    # Calculate the score
    # score = (total number of matches between input and anchor) / (number of anchor imgs)
    score = score / len(anchor_imgs)

    # Determine if the input and anchor are the same face based on threshold
    if score >= threshold:
        print(f"Test image is a {score*100}% match with resident. Unlocking the door! :)")
    else:
        print(f"Test image is a {score*100}% match with resident. This person doesn't belong to this house! >:(")


def doorlock(siamese_model):
    # Initialize the webcam
    cv2.namedWindow('Smart Doorlock')
    cap = cv2.VideoCapture(1)
    start = False
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Crop frame to 720x529, same aspect ratio as model input
        frame = frame[:,376:905, :]

        # Start video
        cv2.imshow('Smart Doorlock', frame)

        # Wait for 's' to be pressed before starting
        if (cv2.waitKey(1) & 0XFF == ord('s')) and not start:
            start = True
            next_capture_time = time.time() + 1

        # Once 's' is pressed, take a photo every 5 seconds
        if start:
            if time.time() >= next_capture_time:
                imgname = os.path.join("test/", '{}.jpg'.format(uuid.uuid1()))
                cv2.imwrite(imgname, frame)
                # Predict whether the photo matches the anchor
                predict_image(imgname, siamese_model)
                next_capture_time = time.time() + 5

        # Stop once 'q' is pressed
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()


def main():
    left_input = Input((62, 47, 3))
    right_input = Input((62, 47, 3))
    output_shape = 62

    # CNN for Siamese Neural Network
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

    # Put both input images through the CNN
    encoded_l = convnet(left_input)
    encoded_r = convnet(right_input)

    # Calculate the L1 distance between both CNN outputs 
    L_layer = L1()
    L1_distance = L_layer([encoded_l, encoded_r])

    # Output a number between 0 and 1 using sigmoid activation
    # where 1 indicates the pair of images is a match, else 0
    prediction = Dense(1,activation='sigmoid')(L1_distance)
    siamese_net = Model(inputs=[left_input,right_input],outputs=prediction)

    # Compile the network using binary crossentropy loss and Adam optimizer
    optimizer = Adam(0.001, decay=2.5e-4)
    siamese_net.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

    # Load the inputs and labels from the faces dataset
    X, n_features, y, target_names, n_classes = load_dataset()

    # Split the inputs into training and testing
    # (we will actually use the test data for validation)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, train_size=0.8)

    # Separate the dataset into a list of images and labels
    image_list = np.split(X_train, X_train.shape[0])
    label_list = np.split(y_train, y_train.shape[0])

    # Create the training dataset
    left_input = []
    right_input = []
    targets = []
    pairs = 6

    # For each input image, pair it with other images to compare to
    for i in range(len(label_list)):
        # Half the pairs should be the same person
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are the same person 
            while compare_to == i or label_list[i] != label_list[compare_to]:
                # Find a random image to compare to
                compare_to = random.randint(0, X_train.shape[0] - 1)
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
            # The predicted output is 1 when the pair of images are the same person
            targets.append(1.)
        # Half the pairs should be different people
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are different people
            while compare_to == i or label_list[i] == label_list[compare_to]:
                compare_to = random.randint(0, X_train.shape[0] - 1)
            left_input.append(image_list[i])
            right_input.append(image_list[compare_to])
            # The predicted output is 0 when the pair of images are different people
            targets.append(0.)
                
    left_input = np.squeeze(np.array(left_input))
    right_input = np.squeeze(np.array(right_input))
    targets = np.squeeze(np.array(targets))

    # Do the same procedure to get testing (but actually validation) data
    test_image_list = np.split(X_test, X_test.shape[0])
    test_label_list = np.split(y_test, y_test.shape[0])

    test_left = []
    test_right = []
    test_targets = []

    for i in range(len(test_label_list)):
        # Half the pairs should be the same person
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are the same person
            while compare_to == i or test_label_list[i] != test_label_list[compare_to]:
                compare_to = random.randint(0, X_test.shape[0] - 1)
            test_left.append(test_image_list[i])
            test_right.append(test_image_list[compare_to])
            # The predicted output is 1 when the pair of images are the same person
            test_targets.append(1.)
        # Half the pairs should be different people
        for _ in range(pairs):
            compare_to = i
            # Make sure it's not comparing to itself and these are different people
            while compare_to == i or test_label_list[i] == test_label_list[compare_to]:
                compare_to = random.randint(0, X_test.shape[0] - 1)
            test_left.append(test_image_list[i])
            test_right.append(test_image_list[compare_to])
            # The predicted output is 0 when the pair of images are different people
            test_targets.append(0.)

    test_left = np.squeeze(np.array(test_left))
    test_right = np.squeeze(np.array(test_right))
    test_targets = np.squeeze(np.array(test_targets))

    # Print a summary of the model architecture
    siamese_net.summary()

    # Train the model on training input and use test data as validation
    # We will test the model performance on our own (new) faces
    siamese_net.fit([left_input,right_input], targets,
            batch_size=16,
            epochs=30,
            verbose=1,
            validation_data=([test_left,test_right],test_targets))
    
    # Run the smart doorlock
    doorlock(siamese_net)

    # # Choose a few pairs of images from test set and visualize model performance
    # print("Let's see how our model does on our favorite pair of test images")
    # img = cv2.imread("datasets/lfw_home/lfw_funneled/Viara_Vike-Freiberga/Viara_Vike-Freiberga_0001.jpg")
    # img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_NEAREST)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img2 = cv2.imread("datasets/lfw_home/lfw_funneled/Zahir_Shah/Zahir_Shah_0001.jpg")
    # img2 = cv2.resize(img2, (47, 62), interpolation=cv2.INTER_NEAREST)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)

    # fig = plt.figure(figsize=(10, 7))
    # rows = 1
    # cols = 2
    # fig.add_subplot(rows, cols, 1)
    # plt.imshow(img)
    # plt.axis('off')
    # plt.title("left image")
    # fig.add_subplot(rows, cols, 2)
    # plt.imshow(img2)
    # plt.axis('off')
    # plt.title("right image")
    # plt.show()

    # # print(test_targets)
    # # print("label: ", test_targets[i])
    # predicted_label = np.round((siamese_net.predict([np.array([img]), np.array([img2])]))[0])
    # print("predicted label: ", predicted_label)

    # # if test_targets[i] == predicted_label:
    # #     print("YAYYYYYYY :))))) WOOOHOOOOO")
    # # else:
    # #     print("FAIL >:(((")
        
    # # Save the model
    # siamese_net.save('siamese_model.keras')
    

if __name__ == "__main__":
    main()