# Flask utils
from flask import Flask, request, render_template
import os
import cv2
import pickle as pkl

# Initialize a Flask app and connect with the server
app = Flask(__name__)

# Process image and predict label
def processImg(IMG_PATH):
    # Load model
    with open("model.pkl", "rb") as f:
        model = pkl.load(f)
    
    # Read and preprocess image
    image = cv2.imread(IMG_PATH)
    image = cv2.resize(image, (28,28))[:,:,1]
    image = image.flatten()

    # Predict label
    predictions = model.predict(image.reshape(1,-1))

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
    
    return render_template("upload.html", predictions=preds, display_image="../img.jpg") 


if __name__ == "__main__":
    app.run()