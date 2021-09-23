'''
    model file : https://github.com/davisking/dlib-models/blob/master/shape_predictor_68_face_landmarks.dat.bz2
'''
from scipy.spatial import distance as sypy_dist     # for distance bet coordinates
from imutils import face_utils                      # for face to np arr of coordinate
import imutils                                      # for resizing
import dlib                                         # for landmarks
import cv2                                          # for image processing
import winsound                                     # for generating sound
import time
# variables ##########################################

frequency = 2500
duration = 800  # ms

drows_count = 0
ear_threshold = 0.3     #  dist betn vertical eye coordinates
ear_frame_threshold = 20    # consecutive eye aspect ratio frame for eye closure

camera_instance = cv2.VideoCapture(0)
face_detector = dlib.get_frontal_face_detector()    # for face detection

landmark_shape_predictor_path = "shape_predictor_68_face_landmarks.dat"
facial_landmark_predictor = dlib.shape_predictor(landmark_shape_predictor_path)   # for facial landmarks

(l_eye_Start, l_eye_End) = face_utils.FACIAL_LANDMARKS_68_IDXS["left_eye"]      # left and right eye location
(r_eye_Start, r_eye_End) = face_utils.FACIAL_LANDMARKS_68_IDXS["right_eye"]     # for np arr slicing

# functions ##########################################
def eye_aspect_ratio(eye):      # we have 6 points for eye  -> 0 | 1,2 | 3 | 4,5
    # vertical ( we can have 2 vertical distance and one horizontal 0-3)
    A = sypy_dist.euclidean(eye[1], eye[5])     # one vertical distance
    B = sypy_dist.euclidean(eye[2], eye[4])     # second vertical distance

    # horizontal
    C = sypy_dist.euclidean(eye[0], eye[3])     # second vertical distance

    ear = (A+B) / (2.0*C)
    # print(A, B, C, ear)
    return ear

# detect drowsiness ####################################

start_time = time.time()
while True:
    _ , frame = camera_instance.read()
    img = imutils.resize(frame, width=450)
    Gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces_rects = face_detector(Gray_img, 0)

    for face_rect in faces_rects:
        # getting 68 landmarks
        face_landmark = facial_landmark_predictor(Gray_img, face_rect)
        face_landmark_np = face_utils.shape_to_np(face_landmark)

        # whole face draw :
        for n in range(0, 68):
            x = face_landmark.part(n).x
            y = face_landmark.part(n).y
            cv2.circle(img, (x, y), 1, (0, 255, 255), 1)

        # getting only eyes landmarks
        left_eye = face_landmark_np[l_eye_Start:l_eye_End]
        right_eye = face_landmark_np[r_eye_Start:r_eye_End]

        # for drawing lines around eye
        left_eye_hull = cv2.convexHull(left_eye)  # for getting all points of left eye
        right_eye_hull = cv2.convexHull(right_eye)
        cv2.drawContours(img, [left_eye_hull], -1, (0, 0, 255), 1)
        cv2.drawContours(img, [right_eye_hull], -1, (0, 0, 255), 1)

        # aspect ratio for both eyes for checking -> eyes closed or open
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(left_eye)

        avg_ear = (left_ear + right_ear) / 2.0

        # checking if eyes are closed
        # print(avg_ear, " vs ", ear_threshold)
        if avg_ear < ear_threshold:
            drows_count += 1
            # print(drows_count)

            # can be a blinking so check for threshold
            current_time = time.time()
            eye_closed_period = current_time-start_time
            print(eye_closed_period)
            if  eye_closed_period > 4:   # sleepy for more than 4 sec
                cv2.putText(img, "Drowsiness Detected", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                winsound.Beep(frequency, duration)
                print(f"     Drowsiness Detected, eyes closed for {eye_closed_period} sec")

        else:   # awake
            drows_count = 0
            start_time = time.time()

    cv2.imshow("Drowsiness Detector", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera_instance.release()
cv2.destroyAllWindows()