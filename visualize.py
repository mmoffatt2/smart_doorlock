from keras import models
import cv2
import os
import numpy as np
from create_model import L1
import matplotlib.pyplot as plt
import uuid
import time

# Load model
siamese_model = models.load_model('siamese_model81.keras', custom_objects={"L1": L1})

# siamese_model.summary()

def predict_image(img_path, threshold=.8):
    score = 0
    img = cv2.imread(img_path)
    img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.imshow(img)
    # plt.show()
    # print(img.shape)
    anchor_path = "anchors/michael"
    anchor_files = os.listdir(anchor_path)
    anchor_imgs = []
    for anchor in anchor_files:
        anchor_img = cv2.imread(f"{anchor_path}/{anchor}")
        anchor_img = cv2.resize(anchor_img, (47, 62), interpolation=cv2.INTER_NEAREST)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        # print(anchor_img.shape)
        anchor_imgs.append(anchor_img)
    # img2 = cv2.imread("datasets/lfw_home/lfw_funneled/Zahir_Shah/Zahir_Shah_0001.jpg")
    # img2 = cv2.resize(img2, (47, 62), interpolation=cv2.INTER_NEAREST)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    # anchor_imgs = []
    # anchor_imgs.append(img2)
    
    for anchor in anchor_imgs:
        # print(anchor.shape)
        # print(img.shape)
        # plt.imshow(anchor)
        # plt.show()
        predicted_label = np.round(siamese_model.predict([np.array([img]), np.array([anchor])])[0])
        # print(predicted_label)
        score += int(predicted_label)

    score = score / len(anchor_imgs)
    if score >= threshold:
        print(f"Test image is a {score*100}% match with resident. Unlocking the door! :)")
        # print(f"Test image is a {score} matches anchor images. Unlocking the door! :)")
    else:
        print(f"Test image is a {score*100}% match with resident. This person doesn't belong to this house! >:(")
        # print("Test image doesn't match anchor images. This person doesn't belong to this house! >:(")

# predict_image("anchors/olivia/oliv.jpg")

def doorlock():
    # Establish a connection to the webcam
    cv2.namedWindow('Smart Doorlock')
    cap = cv2.VideoCapture(1)
    # cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
    # cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

    start = False
    while cap.isOpened():
        ret, frame = cap.read()
    
        # Cut down frame to 720x529
        frame = frame[:,376:905, :]

        # Show video
        cv2.imshow('Smart Doorlock', frame)

        if (cv2.waitKey(1) & 0XFF == ord('s')) and not start:
            start = True
            next_capture_time = time.time() + 1

        if start:
            if time.time() >= next_capture_time:
                # Create the unique file path 
                imgname = os.path.join("test/", '{}.jpg'.format(uuid.uuid1()))
                # Write out anchor image
                cv2.imwrite(imgname, frame)

                predict_image(imgname)

                next_capture_time = time.time() + 5

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