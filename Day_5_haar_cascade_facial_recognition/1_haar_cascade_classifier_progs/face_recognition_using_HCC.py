# Hackathon Simple code
import cv2

algo_xml_file = "haarcascade_frontalface_default.xml"      # in same directory
haar_cascade_facial_classifier = cv2.CascadeClassifier(algo_xml_file)

camera_instance = cv2.VideoCapture(0)

while True:
    img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # img_from_cam = camera_instance.read()[1]         : only for getting img not flag

    if img_flag:
        # resized_original_img = imutils.resize(img_from_cam, width=500)  # 500

        gray_img = cv2.cvtColor(img_from_cam, cv2.COLOR_BGR2GRAY)  # for converting B/W ( binary img)
        face_cordinates = haar_cascade_facial_classifier.detectMultiScale(gray_img, 1.3, 4)

        for (x,y,w,h) in face_cordinates:
            cv2.rectangle(img_from_cam, (x, y), (x+w, y+h), (0,255,0), 2)
        cv2.imshow("Face Detection", img_from_cam)

        key = cv2.waitKey(1)
        if key == 27:
            break


cv2.destroyAllWindows()  # destroying any window if left opened
camera_instance.release()  # releasing camera instance ( releasing instances is good practice )
# realising the camera instance is must .
print("done !")