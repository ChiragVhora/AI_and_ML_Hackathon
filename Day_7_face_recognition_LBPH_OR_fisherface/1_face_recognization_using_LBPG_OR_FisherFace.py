import time

import cv2, numpy, os

###################### face img collect ? ###############################

# if collecting face then turn this variable into true
import numpy

algo_xml_file = "haarcascade_frontalface_default.xml"      # in same directory
haar_cascade_facial_classifier = cv2.CascadeClassifier(algo_xml_file)
scale = 1.3
neighbour = 4
face_width = 150
face_Height = 150
img_folder_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_7_face_recognition_LBPH_OR_fisherface\data"  # raw string

#######################  Setting  ##############################

print("Training....")
(images, labels, names) = ([], [], {})    # ie. (images, labels, names) = 1.png-0-elon, 2.png-0-elon ...

for (subDirs, Dir, Files) in os.walk(img_folder_path):      # traversing data folder and getting all necessary data into (images, labels, names)
    id = 0  # taken as person id, every folder/person will have unique id
    for subDir in Dir:
        names[id] = subDir
        subDir_path = os.path.join(img_folder_path, subDir)
        for img_file_name in os.listdir(subDir_path):
            img_path = os.path.join(subDir_path, img_file_name)
            labels.append(id)
            images.append(cv2.imread(img_path, 0))

        id+=1   # id for second person/folder

(images, labels) = [numpy.array(lst) for lst in [images, labels]]   # converting list to numpy arrays for model training

# model = cv2.face.FisherFaceRecognizer_create()
model = cv2.face.LBPHFaceRecognizer_create()

model.train(images, labels)     # passing numpy arrays for training
print("Training finished.")

camera_instance = cv2.VideoCapture(0)   # for camera frame

unk_count =0
while True:
    success, img = camera_instance.read()
    shape = (500, 500)  # W and H for low computational overhead
    resized_img = cv2.resize(img, shape)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    # obtaining gray img

    faces_locations_in_gray = haar_cascade_facial_classifier.detectMultiScale(gray_img, scale, neighbour)     # detecting face locations
    for (x,y,w,h) in faces_locations_in_gray:
        # drawing rectangle
        cv2.rectangle(resized_img, (x,y), (x+w, y+h), (255, 255, 0), 3)

        # cropping face from gray img
        face_img = gray_img[y:y+h, x:x+w]

        # resize cropped face to certain shape
        resized_face_img = cv2.resize(face_img, (face_width, face_Height))

        # make prediction using cropped face from frame
        prediction = model.predict(resized_face_img)

        #based on prediction alue show if matched or not
        # low prediction[1] value means more accurate match
        text_to_put_on_img = ""
        if prediction[1] < 300:     # reduce 300 to more restrict/accurate matching -> high chances to unknown
            text_to_put_on_img = f"{names[prediction[0]]} mathced : {prediction[1]}"
            unk_count = 0
        else:
            text_to_put_on_img = "Unknown"
            if unk_count > 100:     # for checking
                print("Unknown person : > 100")
                time_ = time.time()
                UK_face_img = resized_img[y:y + h, x:x + w]
                cv2.imwrite("unknown"+str(time_), UK_face_img)

        print(text_to_put_on_img)
        cv2.putText(resized_img, text_to_put_on_img, (x-10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255,0), 2)

        if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
            break
    # showing img
    cv2.imshow("Face recognizer", resized_img)

    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
        break


cv2.destroyAllWindows()  # destroying any window if left opened
camera_instance.release()  # releasing camera instance ( releasing instances is good practice )
# realising the camera instance is must .
print("done !")