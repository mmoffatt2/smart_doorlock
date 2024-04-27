from keras import models
import cv2
import os
import numpy as np
from create_model import L1
import matplotlib.pyplot as plt

# Load model
siamese_model = models.load_model('siamese_model.keras', custom_objects={"L1": L1})

siamese_model.summary()

def predict_image(img_path):
    avg = 0
    img = cv2.imread(img_path)
    img = cv2.resize(img, (47, 62), interpolation=cv2.INTER_NEAREST)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.show()
    # print(img.shape)
    anchor_path = "anchors/michael"
    anchor_files = os.listdir(anchor_path)
    threshold = len(anchor_files)/2
    anchor_imgs = []
    for anchor in anchor_files:
        anchor_img = cv2.imread(f"{anchor_path}/{anchor}")
        anchor_img = cv2.resize(anchor_img, (47, 62), interpolation=cv2.INTER_NEAREST)
        anchor_img = cv2.cvtColor(anchor_img, cv2.COLOR_BGR2RGB)
        # print(anchor_img.shape)
        anchor_imgs.append(anchor_img)
    
    for anchor in anchor_imgs:
        print(anchor.shape)
        print(img.shape)
        predicted_label = np.round(siamese_model.predict([np.array([img]), np.array([anchor])]))
        print(predicted_label)
        avg += predicted_label


    if avg > threshold:
        print("Test image matches anchor images. Unlocking the door!")
    else:
        print("Test image doesn't match anchor images. This person doesn't belong to this house!")

predict_image("anchors/olivia/oliv.jpg")