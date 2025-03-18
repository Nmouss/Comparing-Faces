# imports
from ultralytics import YOLO
import cv2
import os
import tensorflow
import random
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np

class buildTree:
    def __init__(self, hyperplane = None, value = None):
        self.hyperplane = hyperplane
        self.value = value
        self.left = None
        self.right = None
        self.leftVectors = None
        self.rightVectors = None
        self.vectors = None
    
    def findHyperplane(self, vector1, vector2): # creating a plane for the decision tree
    
        normal_vector = vector1 - vector2

        mid_point = (vector1 + vector2) / 2

        constant_coefficient = np.dot(mid_point, normal_vector) # calc III logic lol

        return normal_vector, constant_coefficient
    
    def vector_side(self, normal_vector, constant_coefficient, vector):
        if np.dot(normal_vector, vector) + constant_coefficient < 0:
            return "l"
        else:
            return "r"
    
    def buildTree(self, inputVectors):
        if len(inputVectors) == 0:
            return None
        
        if len(inputVectors) <= 3:
            self.vectors = inputVectors
            return self
        
        vector1 = random.randrange(0, len(inputVectors)) # indices 
        vector2 = random.randrange(0, len(inputVectors))

        while vector1 == vector2:
            vector2 = random.randrange(0, len(inputVectors))
        
        self.hyperplane, self.value = self.findHyperplane(inputVectors[vector1], inputVectors[vector2])

        left, right = [], []

        for vec in inputVectors:
            if self.vector_side(self.hyperplane, self.value, vec) == "l":
                left.append(vec)
            else:
                right.append(vec)

        self.rightVectors = right
        self.leftVectors = left

        if len(left) == len(inputVectors) or len(right) == len(inputVectors): # stop recusion of vectors went to one side (i.e no split)
            self.vectors = inputVectors
            return self
        
        if left: # create the sides when needed (i.e left/right list not none)
            self.left = self.buildTree(left)
        if right:
            self.right = self.buildTree(right)

        return self
    
    def treeTraversal(self, inputVector):
        return self.treeTraversalHelper(inputVector, self)
        

    def treeTraversalHelper(self, inputVector, tree):
        if tree is None: 
            return None

        side = tree.vector_side(tree.hyperplane, tree.value, inputVector)

        if side == "l":
            if len(tree.leftVectors) == 0 or tree.left is None or tree.left == tree: # base cases
                return tree.vectors
            else:
                return self.treeTraversalHelper(inputVector, tree.left)  # Traverse left
        else:
            if len(tree.rightVectors) == 0 or tree.right is None or tree.right == tree: # base cases
                return tree.vectors
            else:
                return self.treeTraversalHelper(inputVector, tree.right)  # Traverse right


face_detection = YOLO("PATH/TO/YOUR/DIRECTORY/yolov8m-face.pt")
image_list = []
for filename in os.listdir("PATH/TO/YOUR/DIRECTORY/"):
    if filename.__contains__(".DS_Store"): # this file was getting into directory for some reason lol
        continue
    else: image_list.append("PATH/TO/YOUR/DIRECTORY/" + filename)

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
    cv2.imwrite(f"PATH/TO/YOUR/DIRECTORY/face{num}.jpg", img)
    num += 1

# cv2.imshow("face image", img) # just showing the image for debugging
# k = cv2.waitKey(0)
    
"""
I am creating the model to get the embedding of the face
"""

embeddingsToFace = {}
model = ResNet50(include_top=False, weights='imagenet', pooling = "avg") # no classification and pooling to average to reduce dimentionality
for face in os.listdir("PATH/TO/YOUR/DIRECTORY/cropped faces"):
    if face.__contains__(".DS_Store"): # this was getting into my array for some reason
        continue
    else:
        face_matrix = tensorflow.keras.preprocessing.image.load_img(f"PATH/TO/YOUR/DIRECTORY/{face}", target_size=(224, 224)) # I am loading the cropped face image as a matrix (224, 224, 3) size x RGB
    
        cropped_face_matrix = np.expand_dims(face_matrix, axis=0) # adds (1, 224, 224, 3) since 1 per batech
        
        finalCropped = preprocess_input(cropped_face_matrix) # Processing to input into the ResNET50 model

        embedding = model.predict(finalCropped) # getting the embedding

        embedding = embedding.flatten() # converts into a vector instead of horizontal 

        embeddingsToFace.setdefault(face, embedding)

#print(embeddingsToFace) # now that I have my vectors time for the decision tree lets go man

def buildDecisionTree():
    faceEmbeddingVectors = list(embeddingsToFace.values())
    Tree = buildTree()
    Tree.buildTree(faceEmbeddingVectors)
    return Tree

def input_image(image, decisionTree):
    resizeX = 224
    resizeY = 224
    resize = (resizeX, resizeY)

    # the input image we want to find the closest to
    image = cv2.imread(image, cv2.IMREAD_COLOR)
    if image is None:
        raise Exception("The image was not found.")
    
    result = face_detection.predict(image, max_det = 1)
    
    result = result[0].boxes
    corners = []
    for coord in result.xyxy[0]: # extracting the coordinates
        corners.append(coord)

    img = image[int(corners[1]):int(corners[3]), int(corners[0]):int(corners[2])] # [y:y+h, x:x+w]
    img = cv2.resize(img, resize, interpolation= cv2.INTER_LINEAR) # resizing
    cv2.imwrite(f"PATH/TO/YOUR/DIRECTORY/croppedInput.jpg", img)

    # getting the embedding
    face_matrix = tensorflow.keras.preprocessing.image.load_img(f"/PATH/TO/YOUR/DIRECTORY/croppedInput.jpg", target_size=(224, 224)) # I am loading the cropped face image as a matrix (224, 224, 3) size x RGB

    cropped_face_matrix = np.expand_dims(face_matrix, axis=0) # adds (1, 224, 224, 3) since 1 per batech
    
    finalCropped = preprocess_input(cropped_face_matrix) # Processing to input into the ResNET50 model

    embedding = model.predict(finalCropped) # getting the embedding

    finalEmbedding = embedding.flatten() # converts into a vector instead of horizontal 

    vectors = Tree.treeTraversal(finalEmbedding)

    return vectors
    

Tree = buildDecisionTree()
image = "PATH/TO/YOUR/DIRECTORY/input.jpeg"
closestVectors = input_image(image, Tree)
print(closestVectors)
# so we can look up in O(1)
closestVectorsSet = {tuple(vec) for vec in closestVectors}
for key, embedding in embeddingsToFace.items():
    if any(np.allclose(embedding, np.array(vec)) for vec in closestVectorsSet):     # fixes floating point precision issues
        image_path = f"PATH/TO/YOUR/DIRECTORY/cropped faces/{key}"

        if not os.path.exists(image_path): # file not found
            print(f"Error: Image not found at {image_path}")
            continue

        img = cv2.imread(image_path)

        if img is None:
            print(f"Error: Could not load image {image_path}")
            continue

        cv2.imshow("Face", img)
        cv2.waitKey(0)  
        cv2.destroyAllWindows()


