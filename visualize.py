from keras import models
import cv2
import os
import numpy as np
from create_model import L1
import matplotlib.pyplot as plt
import uuid
import time

# Load model
siamese_model = models.load_model('siamese_model.keras', custom_objects={"L1": L1})

# siamese_model.summary()

def predict_image(img_path, threshold=.7):
    score = 0
    img = cv2.imread(img_path)
    img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # print(img.shape)
    anchor_path = "anchors/michael"
    anchor_files = os.listdir(anchor_path)
    threshold = len(anchor_files)*.7
    anchor_imgs = []
    for anchor in anchor_files:
        anchor_img = cv2.imread(f"{anchor_path}/{anchor}")
        anchor_img = cv2.resize(anchor_img, (47, 62), interpolation=cv2.INTER_NEAREST)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        # print(anchor_img.shape)
        anchor_imgs.append(anchor_img)
    
    for anchor in anchor_imgs:
        # print(anchor.shape)
        # print(img.shape)
        predicted_label = np.round(siamese_model.predict([np.array([img]), np.array([anchor])]))
        # print(predicted_label)
        score += predicted_label

    if score > threshold:
        print("Test image matches anchor images. Unlocking the door! :)")
    else:
        print("Test image doesn't match anchor images. This person doesn't belong to this house! >:(")

# predict_image("anchors/olivia/oliv.jpg")

def doorlock():
    # Establish a connection to the webcam
    cv2.namedWindow('Smart Doorlock')
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    next_capture_time = time.time() + 10

    while cap.isOpened(): 
        ret, frame = cap.read()
    
        # Cut down frame to 720x529
        frame = frame[:,376:905, :]
        
        # Show video
        cv2.imshow('Smart Doorlock', frame)

        if time.time() >= next_capture_time:
            # Create the unique file path 
            imgname = os.path.join("test/olivia/", '{}.jpg'.format(uuid.uuid1()))

            # Write out anchor image
            cv2.imwrite(imgname, frame)

            predict_image(imgname)

            next_capture_time = time.time() + 10

        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break
            
    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


def main():
    doorlock()


if __name__ == "__main__":
    main()