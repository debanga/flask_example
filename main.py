# Flask utils
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os

from sklearn.metrics import accuracy_score,confusion_matrix # metrics error
from sklearn.model_selection import train_test_split # resampling method
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle


# Define a flask app
app = Flask(__name__)

# Process image and predict label
def processImg(IMG_PATH):
    with open("model.pkl", "rb") as f:
        knn = pickle.load(f)
    
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (28,28))[:,:,1]
    print(image.shape)
    image = image.flatten()
    predictions = knn.predict(image.reshape(1,-1))

    return list(map(int, list(predictions)))

@app.route("/")
def index():
    return render_template("index.html")


@app.route('/uploader', methods = ['POST'])
def upload_file():
    predictions=""

    if request.method == 'POST':
        f = request.files['file']

        f.save("static/img.jpg")
        preds = processImg("static/img.jpg")

        print("preds:::",preds)
    return render_template("upload.html", predictions=preds, display_image="../img.jpg") 


if __name__ == "__main__":
    app.run(host="0.0.0.0",debug=True,port="4100")