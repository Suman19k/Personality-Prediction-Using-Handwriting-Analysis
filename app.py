from flask import Flask, render_template, request, redirect, url_for, session
import re
import os
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import matplotlib.pyplot as plt
import numpy as np
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
from csv import writer
import pandas as pd
from flask_material import Material
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Input, Lambda, Dense, Flatten,Dropout,Conv2D,MaxPooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import pickle
from tensorflow.keras.models import load_model
UPLOAD_FOLDER = 'static/uploads/'

# EDA PKg
import pandas as pd 
import numpy as np 

# ML Pkg


app = Flask(__name__)
Material(app)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])
class_names = ['Anxious', 'Cooperative', 'Enthusiastic', 'Openness', 'Responsible', 'System image']
class_name1=['Non Valid', 'Valid']
img_height = 300
img_width = 300
def allowed_file(filename):
	return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Enter your database connection details below

@app.route('/')
def index():
    return render_template("login.html")

@app.route('/home')
def home():
    return render_template('index.html')
    # User is not loggedin redirect to login page

@app.route('/about')
def about():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('about.html')
    # User is not loggedin redirect to login page

@app.route('/contact')
def contact():
    # Check if user is loggedin
        
        # User is loggedin show them the home page
    return render_template('contact.html')

@app.route('/',methods=['GET', 'POST'])
def login():
    msg = ''
    # Check if "username" and "password" POST requests exist (user submitted form)
    if request.method == 'POST' and 'username' in request.form and 'password' in request.form:
        # Create variables for easy access
        username = request.form['username']
        password = request.form['password']
        
                # If account exists in accounts table in out database
        if username=="admin" and password=="admin":
            # Create session data, we can access this data in other routes
            # Redirect to home page
            return render_template('index.html')
        else:
            # Account doesnt exist or username/password incorrect
            msg = 'Incorrect username/password!'
    return render_template('login.html', msg=msg)




@app.route('/upload_image',methods=["POST"])
def upload_image():
	
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    print(file)
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        num_classes = 2

        model = Sequential([
        layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dense(num_classes)
        ])
        model.compile(optimizer='adam',
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

        model.load_weights("personalityValid.h5")
        test_data_path = path

        img = keras.preprocessing.image.load_img(
            test_data_path, target_size=(img_height, img_width)
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        print("classname==========================================",class_name1[np.argmax(score)])

        if class_name1[np.argmax(score)]=='Non Valid':
             return render_template('contact.html',res=1, aclass=class_name1[np.argmax(score)])
        else:
            model = Sequential([
            layers.experimental.preprocessing.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
            layers.Conv2D(16, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(32, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding='same', activation='relu'),
            layers.MaxPooling2D(),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(6)
            ])
            model.compile(optimizer='adam',loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),metrics=['accuracy'])

            model.load_weights("personalityhandwriting.h5")

            test_data_path = path

            img = keras.preprocessing.image.load_img(
            test_data_path, target_size=(img_height, img_width)
            )
            img_array = keras.preprocessing.image.img_to_array(img)
            img_array = tf.expand_dims(img_array, 0) # Create a batch

            predictions = model.predict(img_array)
            score = tf.nn.softmax(predictions[0])
            probabilities= tf.nn.softmax(predictions[0])

            print(probabilities)
            

            for i,prob in enumerate(probabilities):
                 print("class {}: {:.2f}% confidence".format(i,100*prob))

            print(
            "This image most likely belongs to {} with a {:.2f} percent confidence."
            .format(np.argmax(score), 100 * np.max(score))
        )
            print(np.argmax(score))
            if class_names[np.argmax(score)]=="System image":
                return render_template('contact.html',aclass=class_names[np.argmax(score)],res=2)
            else:
                 return render_template('contact.html',aclass=class_names[np.argmax(score)],ascore=round(100 * np.max(score),2),res=3,filename=filename,class_names=class_names,probabilities=probabilities)


@app.route('/display/<filename>')
def display_image(filename):
	#print('display_image filename: ' + filename)
	return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
	app.run(debug=True)
