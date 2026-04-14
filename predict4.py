import cv2
import numpy as np
import tensorflow as tf

def predict_image(path):
    model = tf.keras.models.load_model("models4/model.h5")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (256,256))
    img = img / 255.0

    img = img.reshape(1,256,256,1)

    prediction = model.predict(img)[0][0]

    if prediction > 0.5:
        print("🦠 PNEUMONIA DETECTED")
    else:
        print("✅ NORMAL")