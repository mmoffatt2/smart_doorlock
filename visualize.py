from keras import models
import cv2
import os
import numpy as np

# Load model
siamese_model = models.load_model('siamese_model_77.keras')

siamese_model.summary()

def predict_image(img_path):
    avg = 0
    img = cv2.imread(img_path)
    anchor_path = "achors/michael"
    anchor_files = os.listdir(anchor_path)
    anchor_imgs = []
    for anchor in anchor_files:
        anchor_img = cv2.imread(anchor)
        anchor_imgs.append(anchor_img)
    
    for anchor in anchor_imgs:
        predicted_label = np.round(siamese_model.predict([img, anchor]))
        avg += predicted_label

    if avg > 100:
        print("Test image matches anchor images. Unlocking the door!")
    else:
        print("Test image doesn't match anchor images. This person doesn't belong to this house!")
