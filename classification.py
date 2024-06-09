# app.py
import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from werkzeug.utils import secure_filename
from keras.applications import vgg16
from tensorflow.keras.models import Sequential
from keras.layers import Dense
import numpy as np
import cv2
import uvicorn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]
IMG_SIZE = 224
IMAGE_SIZE = (IMG_SIZE, IMG_SIZE)
UPLOAD_FOLDER = 'uploads'

app = FastAPI()

vgg16_model = vgg16.VGG16()
model1 = Sequential()

for layer in vgg16_model.layers[:-3]:
    model1.add(layer)

for layer in model1.layers:
    layer.trainable = False

model1.add(Dense(4, activation='softmax'))
model1.load_weights("model/model.h5")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prepare(filepath):
    img_array = cv2.imread(filepath, cv2.IMREAD_COLOR)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)


def predict(file_path):
    prediction = model1.predict([prepare(file_path)])

    if np.argmax(prediction) == 0:
        output = "glioma_tumor"
    elif np.argmax(prediction) == 1:
        output = "meningioma_tumor"
    elif np.argmax(prediction) == 2:
        output = "no_tumor"
    elif np.argmax(prediction) == 3:
        output = "pituitary_tumor"
    return output


@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    if not allowed_file(file.filename):
        raise HTTPException(status_code=400, detail="Invalid file type")
    
    file_path = os.path.join(UPLOAD_FOLDER, secure_filename(file.filename))
    with open(file_path, "wb") as buffer:
        buffer.write(file.file.read())
    
    result = predict(file_path)
    return JSONResponse(content={"prediction": result})

# if __name__ == "__main__":
#     
#     uvicorn.run(app)
