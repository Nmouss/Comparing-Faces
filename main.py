# imports
from ultralytics import YOLO
import cv2
import os
import tensorflow
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
"""
I am creating the input image obj, and face detection obj
This code will crop faces listed in the directory and append them to a
new directory. It crops only the face using a YOLO model.
"""

face_detection = YOLO("/PATH/TO/YOUR/DIRECTORY/yolov8m-face.pt")
image_list = []
for filename in os.listdir("/PATH/TO/YOUR/DIRECTORY"):
    if filename.__contains__(".DS_Store"): # this file was getting into directory for some reason lol
        continue
    else: image_list.append("/PATH/TO/YOUR/DIRECTORY" + filename)

num = 0
resizeX = 224
resizeY = 224
resize = (resizeX, resizeY)
for image in image_list:
    input_image = cv2.imread(image)
    if input_image is None:
        raise Exception("Cannot read the image ", image)
    results = face_detection.predict(input_image, max_det = 1) # making sure that we only have 1 face
    boxes = results[0].boxes # extracting the bounding boxes coordinates
    corners = []
    for coord in boxes.xyxy[0]: # extracting the coordinates
        corners.append(coord)

    img = input_image[int(corners[1]):int(corners[3]), int(corners[0]):int(corners[2])] # [y:y+h, x:x+w]
    img = cv2.resize(img, resize, interpolation= cv2.INTER_LINEAR) # resizing
    cv2.imwrite(f"/PATH/TO/YOUR/DIRECTORY/face{num}.jpg", img)
    num += 1

# cv2.imshow("face image", img) # just showing the image for debugging
# k = cv2.waitKey(0)
    
"""
I am creating the model to get the embedding of the face
"""
embeddingsToFace = {}
model = ResNet50(include_top=False, weights='imagenet', pooling = "avg") # no classification and pooling to average to reduce dimentionality
for face in os.listdir("/PATH/TO/YOUR/DIRECTORY"):
    if face.__contains__(".DS_Store"): # this was getting into my array for some reason
        continue
    else:
        print(face)
        face_matrix = tensorflow.keras.preprocessing.image.load_img(f"/PATH/TO/YOUR/DIRECTORY/cropped faces/{face}", target_size=(224, 224)) # I am loading the cropped face image as a matrix (224, 224, 3) size x RGB
    
        cropped_face_matrix = np.expand_dims(face_matrix, axis=0) # adds (1, 224, 224, 3) since 1 per batech
        
        finalCropped = preprocess_input(cropped_face_matrix) # Processing to input into the ResNET50 model

        embedding = model.predict(finalCropped) # getting the embedding

        embeddingsToFace.setdefault(face, embedding)

print(embeddingsToFace) # now that I have my vectors time for the decision tree lets go man
