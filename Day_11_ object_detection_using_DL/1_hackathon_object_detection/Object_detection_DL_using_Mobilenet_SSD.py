'''
    found the almost same code in YT 21 jan, 2021  (^_^)
        Video tutorial : https://youtu.be/WJc-peqriVc
        model Files are obtained from : https://github.com/PacktWorkshops/The-Computer-Vision-Workshop/tree/master/Chapter07/Exercise7.04
'''

import numpy as np
import imutils
import cv2
import time

prototext_path = "MobileNetSSD_deploy.prototxt.txt"
Mobilenet_SSD_model_path = "MobileNetSSD_deploy.caffemodel"

confidence_threshold = 0.2
# load the pre-trained model using opencv dnn¶ , trained from cafe
mobilenet_ssd_model = cv2.dnn.readNetFromCaffe(prototext_path, Mobilenet_SSD_model_path)

# List of classes that can be classified by MobileNet model
classes =  ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
            "diningtable",  "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
# random color for all classes
colors = np.random.uniform(255, 0, size=(len(classes), 3))

# for camera instance
camera_instance = cv2.VideoCapture(0)
time.sleep(1)   # 1 sec for camera obtaining


while True:
    img_flag, frame_img = camera_instance.read()
    if img_flag:
        # fitting into display
        resized_frame_img = imutils.resize(frame_img, width=500)

        # obtaining height and width
        (h, w) = resized_frame_img.shape[:2]

        # 300x300 required for model
        prerequisite_image_for_model = cv2.resize(resized_frame_img, (300, 300))

        # blob for img
        blob = cv2.dnn.blobFromImage(prerequisite_image_for_model, 0.007843, (300, 300), 127.5)     # img, scaling factor, spacial size, mean Sub- val

        # img testing
        mobilenet_ssd_model.setInput(blob)
        Results_detections = mobilenet_ssd_model.forward()

        results_detections_shape = Results_detections.shape[2]  # no of shapes detected = no of items detected
        obj_in_frame = ""
        for i in np.arange(0, results_detections_shape):    # for all items detected/predicted
            confidence = Results_detections[0, 0, i, 2]     # accuracy of prediction for i th object
            if confidence > 0.5:    # more than 50 % surety
                # get the index of object
                idx = int(Results_detections[0, 0, i, 1])

                # location of object in image
                box = Results_detections[0, 0, i, 3:7] * np.array([w, h, w, h])     # 3:7 = 3,4,5,6
                (startX, startY, endX, endY) = box.astype("int")

                # drawing rectangle and putting text
                cv2.rectangle(resized_frame_img, (startX, startY), (endX, endY), colors[idx], 2)

                # if rectangle start point above the 15 px from top then show text in the rectangle else on the rectangle
                y = startY - 15 if startY - 15 > 15 else startY + 20    # kind of ?: syntax
                startX += 5     # margin of 5 px - x axis

                label = "{}: {:.2f}%".format(classes[idx], confidence * 100)
                cv2.putText(resized_frame_img, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[idx], 2)

                # for printing objects
                obj_in_frame += " "+classes[idx]

    obj_in_frame = " 0 " if obj_in_frame == "" else obj_in_frame
    print(f"objects in the frames are : {obj_in_frame}")
    cv2.imshow("Object detection using mobile net SSD - CNN", resized_frame_img)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera_instance.release()
cv2.destroyAllWindows()

