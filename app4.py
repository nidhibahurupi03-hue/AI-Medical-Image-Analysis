from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

# 👇 IMPORTANT change (templates4 use करत आहोत)
app = Flask(__name__, template_folder='templates4')

# Load model
model = tf.keras.models.load_model("models4/model.h5")

# Home page
@app.route('/')
def home():
    return render_template('index4.html')

# Prediction
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    if file:
        filepath = os.path.join("static", file.filename)
        file.save(filepath)

        # Preprocess
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = img.reshape(1, 256, 256, 1)

        # Predict
        prediction = model.predict(img)[0][0]

        if prediction > 0.5:
            result = "🦠 PNEUMONIA DETECTED"
        else:
            result = "✅ NORMAL"

        return render_template('index4.html', result=result, image=filepath)

    return "Error"

if __name__ == '__main__':
    app.run(debug=True)