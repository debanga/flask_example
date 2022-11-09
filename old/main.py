#%%
from flask import Flask, render_template, request

from sklearn.metrics import accuracy_score,confusion_matrix # metrics error
from sklearn.model_selection import train_test_split # resampling method
from sklearn.datasets import load_digits
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier

import cv2
import pickle

from flask_cors import CORS, cross_origin


#%%
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


# Initializing flask application
app = Flask(__name__)
cors = CORS(app)

@app.route("/")
def main():
    return """
        Application is working
    """

# About page with render template
@app.route("/about")
def postsPage():
    return render_template("about.html")

# Process images
@app.route("/process", methods=["POST"])
def processReq():
    #https://www.kaggle.com/datasets/scolianni/mnistasjpg
    data = request.files["img"]
    data.save("img.jpg")

    resp = processImg("img.jpg")


    return resp


if __name__ == "__main__":
    app.run(debug=True)
