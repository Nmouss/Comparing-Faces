from ultralytics import YOLO
import cv2
import torch
# import tensorflow as tf 
# import keras
"""
I am creating the input image obj, and face detection obj
"""
face_detection = YOLO("yolov8m-face.pt")
input_image = cv2.imread("face.JPG")
if input_image is None:
    raise Exception("Cannot read the image.")
results = face_detection(input_image)
boxes = results[0].boxes # extracting the bounding boxes coordinates
corners = []
for coord in boxes.xyxy: # extracting the coordinates
    corners.append(coord)

img = cv2.rectangle(input_image, (int(corners[0][0]),int(corners[0][1])), (int(corners[0][2]),int(corners[0][3])), -23, 10)


cv2.imshow("face image", img)
k = cv2.waitKey(0)

# add embedding

# model = keras.Sequential()
# model.add(keras.layers.Embedding(1000, 128))
# print(model.summary())
