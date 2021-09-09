'''
    from youtube video
    little modified for saving faces
'''
import cv2
import os
import time

###################### face img collect ? ###############################

# if collecting face then turn this variable into true
only_face_collection = True
algo_xml_file = "haarcascade_frontalface_default.xml"      # in same directory
haar_cascade_facial_classifier = cv2.CascadeClassifier(algo_xml_file)
scale = 1.3
neighbour = 4
face_width = 150
face_Height = 150
save_gray_color_face = False    # false - saves original color , true for gray img

#######################  Setting  ##############################

myPath = 'data/images' # your pics:  '/home/pi/Desktop/data/images'
cameraNo = 0
cameraBrightness = 180
moduleVal = 10  # SAVE EVERY I th FRAME TO AVOID REPETITION
minBlur = 500  # SMALLER VALUE MEANS MORE BLURRINESS PRESENT
grayImage = False # IMAGES SAVED COLORED OR GRAY
saveData = True   # SAVE DATA FLAG
showImage = True  # IMAGE DISPLAY FLAG
imgWidth = 180
imgHeight = 120


#####################################################

global countFolder
cap = cv2.VideoCapture(cameraNo)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10,cameraBrightness)


count = 0
countSave =0

def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists( myPath+ str(countFolder)):
        countFolder += 1
    os.makedirs(myPath + str(countFolder))

if saveData:saveDataFunc()


while True:

    success, img = cap.read()
    img = cv2.resize(img,(imgWidth,imgHeight))
    if grayImage:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    if saveData:
        blur = cv2.Laplacian(img, cv2.CV_64F).var()     # getting how much blurred the img is ,if more blurred then-> discard
        if count % moduleVal ==0 and blur > minBlur:

            # extra code for saving face only (hackathon logic)---------------------------------
            if only_face_collection:
                gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # for converting B/W ( binary img)
                face_cordinates = haar_cascade_facial_classifier.detectMultiScale(gray_img, scaleFactor=scale, minNeighbors=neighbour)

                for (x, y, w, h) in face_cordinates:
                    face_from_original = img[y:y+h, x:x+w]
                    Cropped_face = face_from_original
                    if save_gray_color_face:
                        Cropped_face = face_from_gray_img = gray_img[y:y+h, x:x+w]

                    resize_face_from_Selected_img = cv2.resize(Cropped_face, (face_width,face_Height))
                    img = resize_face_from_Selected_img
                    nowTime = time.time()
                    cv2.imwrite(myPath + str(countFolder) +
                                '/' + str(countSave) + "_" + str(int(blur)) + "_" + str(nowTime) + ".png", img)
                    countSave += 1

                    if showImage:
                        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.imshow("Reference", img)

                    # hackathon logic Ends here --------------------------------------------
            else:
                nowTime = time.time()
                cv2.imwrite(myPath + str(countFolder) +
                        '/' + str(countSave)+"_"+ str(int(blur))+"_"+str(nowTime)+".png", img)
                countSave+=1
        count += 1

    if showImage:
        cv2.imshow("Image", img)


    if cv2.waitKey(1) & 0xFF == ord('q') or cv2.waitKey(1) == 27:
        break

cap.release()
cv2.destroyAllWindows()
