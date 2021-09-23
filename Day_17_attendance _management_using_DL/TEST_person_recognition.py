import numpy as np
import imutils
import pickle
import time
import cv2

embedding_model_path = "model/openface.nn4.small2.v1.t7"

embedding_file = "output/embeddings.pickle"     # contain all name and embeddings
recognizer_file = "output/recognizer.pickle"    # contain trained model
LabelEnc_file = "output/label_encoder.pickle"      # contain encoded names
conf = 0.5

print("Loading face detector...")
prototext = "model/deploy.prototxt"
model = "model/res10_300x300_ssd_iter_140000.caffemodel"
detector_net = cv2.dnn.readNetFromCaffe(prototext, model)

print("Loading face recognizer ...")
embedder = cv2.dnn.readNetFromTorch(embedding_model_path)

recognizer = pickle.load(open(recognizer_file, "rb").read())
label_encoder = pickle.load(open(LabelEnc_file, "rb").read())

# done
print("Video Streaming ...")
camera_instance = cv2.VideoCapture(0)
time.sleep(2)

while True:
    _, img = camera_instance.read()
    frame = imutils.resize(img, width=600)
    (h, w) = frame.shape[:2]
    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

    detector_net.setInput(image_blob)
    detections = detector_net.forward()

    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > conf:
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
            embedder.setInput(face_image_blob)
            vec = embedder.forward()

            # predictions
            preds = recognizer.predict_proba(vec)[0]
            j = np.argmax(preds)
            proba = preds[j]
            name = label_encoder.classes_[j]

            text = "{} : {:.2f}%".format(name, proba*100)
            y = startY - 10 if startY - 10 > 10 else startY + 10
            cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
            cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

    cv2.imshow("Recognizer", frame)
    key = cv2.waitKey(1)
    if key == 27:
        break

camera_instance.release()
cv2.destroyAllWindows()