from flask import Flask, render_template, request
from PIL import Image
# from tensorflow.keras.preprocessing.image import load_img
import io
#import cv2
import tensorflow as tf
# from tensorflow.keras import datasets
# from keras.applications.imagenet_utils import preprocess_input
# from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
# from tensorflow.keras.models import Sequential


import numpy as np


shape_ = (32,32,3)
out = {1:"Truck",0:"Ship"}
load_model = tf.keras.models.load_model("shiptruck_model")

app = Flask(__name__)

@app.route("/", methods=['GET', 'POST'])
def main():
    return render_template("index.html")

@app.route("/submit",methods=['GET', 'POST'])
def predict():
    print("hello")
    if request.method == 'POST':
        img_requested = request.files['file'].read()
        img = Image.open(io.BytesIO(img_requested))
        
        print(type(img))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = img.resize((32, 32))
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        img = img.astype('float32')
        img = img / 255.0
      
       
       
        pred = load_model.predict(img)
        print(np.argmax(pred))
    return render_template("index.html", prediction = out[np.argmax(pred)]) 
if __name__ == '__main__':
    app.run(port=8080,debug=True)