'''
    All data are pickled together in one file
    Logic:
        1. get all images
        2. get embeddings for face and append it to list of student object
        3. save data into pickle file

    reference : cnn caffe program + model
        https://github.com/gopinath-balu/computer_vision/tree/master/CAFFE_DNN
'''
import os
import pickle

import cv2
import imutils
from imutils import paths
import numpy as np

# variables #############################################
dataset = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_7_face_recognition_LBPH_OR_fisherface\data"
algo_xml_file = "haarcascade_frontalface_default.xml"      # in same directory
haar_cascade_facial_classifier = cv2.CascadeClassifier(algo_xml_file)

prototext = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
embedding_model_path = "model/openface.nn4.small2.v1.t7"
detector_net = cv2.dnn.readNetFromCaffe(prototext, model)

scale = 1.3
neighbour = 4

# imgWidth = 200
# imgHeight = 200

# function #############################################33
# def get_face_embeddings_from_cam():
#     camera_instance = cv2.VideoCapture(0)
#     _, img = camera_instance.read()
#     gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # for converting B/W ( binary img)
#     face_cordinates = haar_cascade_facial_classifier.detectMultiScale(gray_img, scaleFactor=scale, minNeighbors=neighbour)
#
#     if len(face_cordinates) == 1:     # we have face in image
#
#
#     elif len(face_cordinates)>1:
#         print("image have multiple faces ! , require only one .")
#     else:
#         print("no face found !")
#
#     return None


# def get_student_data():
#     name = input("Enter the Name of Student : ")
#     roll_no = input("Roll no : ")
#     return name, roll_no

#########################################################3
Known_embeddings = []
Known_names = []
total=0
conf = 0.5

# getting image path
image_paths = list(paths.list_images(dataset))

# read image one by one and apply face detection
for (i, imagePath) in enumerate(image_paths):
    name = imagePath.split(os.path.sep)[-2]     # folder name as name
    print(f"processing image {i}/{len(image_paths)}")
    img = cv2.imread(imagePath)
    img = imutils.resize(img, width=600)
    (h, w) = img.shape[:2]

    # converting image to blob for face detection
    image_blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    # caffe model for face detection
    detector_net.setInput(image_blob)   # set the blob
    detections = detector_net.forward()     # getting all face detection

    if len(detections)>0:   # we have face
        i = np.argmax(detections[0, 0, :, 2])
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
            # getting cordinates of face
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])  # 3:7 = 3,4,5,6
            (startX, startY, endX, endY) = box.astype("int")
            # cut the face part
            face_img = img[startY:endY, startX:endX]
            fH, fW = face_img.shape[:2]

            if fH < 20 and fW < 20 :    # face too small for training
                continue

            # blob image for face
            face_image_blob = cv2.dnn.blobFromImage(face_img, 1.0/255, (300, 300), (104.0, 177.0, 123.0))
            # getting the embeddings for face
            embedding_model = cv2.dnn.readNetFromTorch(embedding_model_path)
            embedding_model.setInput(face_image_blob)
            vec = embedding_model.forward()
            Known_names.append(name)
            Known_embeddings.append(vec.flatten())
            total+=1
        pass

data = {"embeddings": Known_embeddings, "names":Known_names}

embedding_file = "output/embeddings.pickle"
with open(embedding_file, "eb") as f:
    f.write(pickle.dumps(data))

print("process completed , data saved.")


