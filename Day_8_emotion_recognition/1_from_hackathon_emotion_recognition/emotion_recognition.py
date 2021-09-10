'''
    install facial_emotion_recognition,torch
        and put `cpu` in serialization file as shown in hackathon path looks like : Programs\Python\Python39\Lib\site-packages\torch\serialization
        changes:
            # def load(f, map_location=None, pickle_module=pickle, **pickle_load_args):
            def load(f, map_location='cpu', pickle_module=pickle, **pickle_load_args):
'''
from facial_emotion_recognition import EmotionRecognition
import cv2

emotion_recognition_instance = EmotionRecognition(device="cpu")
camera_instance = cv2.VideoCapture(0)

while True:
    img_flag, frame = camera_instance.read()
    # face rec, emotion rec. , put rectangle , put emotion text :: all in one
    result_frame = emotion_recognition_instance.recognise_emotion(frame, return_type="BGR")     # all things are done by this function
    cv2.imshow("Emotion Recognizer", result_frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

camera_instance.release()
cv2.destroyAllWindows()
print("done!")