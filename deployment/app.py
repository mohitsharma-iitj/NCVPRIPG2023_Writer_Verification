from __future__ import division, print_function
# coding=utf-8
import sys
import os
# import glob
import re
import numpy as np
import tensorflow as tf
import test as test
import tensorflow.keras.backend as K
#

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

model2 = tf.keras.models.load_model("model_simple.h5")
model2.summary()
# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH = 'models/model_resnet.h5'

# # Load your trained model
# model = load_model(MODEL_PATH)
# model._make_predict_function()          # Necessary
# print('Model loaded. Start serving...')

# You can also use pretrained model from Keras
# Check https://keras.io/applications/
#from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model.save('')
print('Model loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))

    # Preprocessing the image
    x = image.img_to_array(img)
    # x = np.true_divide(x, 255)
    x = np.expand_dims(x, axis=0)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    x = preprocess_input(x, mode='caffe')

    preds = model.predict(x)
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']
        f2 = request.files['file2']
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path1 = os.path.join(
            basepath, 'uploads', secure_filename(request.files['file'].filename))
        f.save(file_path1)
        file_path2 = os.path.join(
            basepath, 'uploads', secure_filename(request.files['file2'].filename))
        f2.save(file_path2)
    
        # Make prediction

        preds = test.prediction2(model2,file_path1.replace(os.sep, '/'),file_path2.replace(os.sep, '/'))

        # Process your result for human
        # pred_class = preds.argmax(axis=-1)            # Simple argmax
        # pred_class = decode_predictions(preds, top=1)   # ImageNet Decode
        result = str(preds)               # Convert to string
        return result
    return None


if __name__ == '__main__':
    app.run(debug=True)

