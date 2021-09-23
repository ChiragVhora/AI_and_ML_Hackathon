# HSVtunerHere
import tkinter

# Body
from tkinter import *
from tkinter import font
from tkinter.filedialog import askopenfilename

import cv2

from keras.preprocessing.image import image
import numpy as np
import os

from common_python_files.save_OR_load_model_into_json import cmn_load_nn_model



# function

# variables ##########################################
img_file_path = ""
#######################################################


def on_Brows(*args):
    global test_image
    global img_file_path
    img_file_path = askopenfilename(title="Select An Image to classify", filetypes=(
        ("jpeg files", "*.jpg"), ("gif files", "*.gif*"), ("png files", "*.png")))

    print(img_file_path)
    from PIL import ImageTk, Image
    from PIL import ImageTk as itk
    photo = itk.PhotoImage(Image.open(img_file_path).resize((450, 450)))
    test_image.configure(image=photo)
    test_image.image = photo
    # photo = itk.PhotoImage(file=img_file_path)
    # chosen_img = PhotoImage(img_file_path)/
    pass

def classify_gujarati_classifier(img_path):

    model_name = "my_hand_gesture_cnn_model"
    cnn_model = cmn_load_nn_model(model_name)
    print("model loaded.")
    target_size = (100, 100, 1)
    classes = ["index finger", "ok gesture", "palm", "pinky finger", "thumb down", "thumb up"]

    global success_count
    # this converts img to 256x256 gray image
    test_image = image.load_img(img_path, target_size=target_size,
                                color_mode="grayscale")  # for the target we trained

    test_image_arr = image.img_to_array(test_image)
    test_image_arr_expanded = np.expand_dims(test_image_arr, axis=0)

    results = cnn_model.predict(test_image_arr_expanded)  # passing image arr as we train for that
    # print(results)

    # print("result : ", results)
    pred_arr = np.argmax(results[0])
    # print(pred_arr)
    # print("arr : ", arr)

    # maxx = np.amax(arr)
    # max_prob = pred_arr.argmax(axis=0)
    # max_prob += 1
    prediction = classes[int(pred_arr)]
    # prediction = classes[max_prob-1]

    # first_3_char = prediction[:3]      # i.e 01_palm -> 01
    # classified_file_3_char = f"{first_3_char}"
    # print(classified_file_3_char)

    image_name = os.path.basename(img_path)
    if prediction in image_name:  # in img name
        print("correct : ", image_name, " identified as ", prediction)
    else:
        print("False : ", image_name, " identified as ", prediction)

    return prediction
    pass


def on_Classify(*arg):
    global prediction_label
    global img_file_path
    prediction_label["text"] = classify_gujarati_classifier(img_file_path)
    pass

def on_Train(*args):
    pass


from PIL import ImageTk, Image

def on_press(key):
    global Flag_cam
    if key == key.enter:
        Flag_cam = False
    pass

def on_Capture(*args):
    global test_image
    global img_file_path
    global Flag_cam
    camera_instance = cv2.VideoCapture(0)
    color_upper_hsv_val = (44, 58, 198)
    color_lower_hsv_val = (0, 30, 89)
    while True:
        _,frame = camera_instance.read()

        # for showing to label
        RGB_img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        img = Image.fromarray(cv2.resize(RGB_img, (450, 450)))
        imgtk = ImageTk.PhotoImage(image=img)
        test_image.configure(image=imgtk)
        test_image.image = imgtk

        # pre-processing for classification
        HSV_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # for converting to HSV img

        # step 3 - returned color ranged objects as white pixels
        only_color_range_objects_img = cv2.inRange(HSV_image, color_lower_hsv_val, color_upper_hsv_val)

        check_tuning_times_operation_perform = 3
        noise_removed_objects_img_dilate = cv2.dilate(only_color_range_objects_img, None,
                                                      iterations=check_tuning_times_operation_perform)  # append white pixels at boundry
        check_tuning_times_operation_perform = 1
        noise_removed_objects_img_erode = cv2.erode(noise_removed_objects_img_dilate, None,
                                                    iterations=check_tuning_times_operation_perform)  # removing extra white pixels

        # showing the reference images
        cv2.imshow("img to capture", frame)
        cv2.imshow("save reference", noise_removed_objects_img_erode)
        '''
            enter any key for getting next image , Enter for selecting the image
            we are looking for white color object so tune HSV value accordingly 
                using these reference images
        '''
        key = cv2.waitKey(0)
        if key == 13:
            cv2.imwrite("HG_capture.jpg", noise_removed_objects_img_erode)
            break


    img_file_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_13_HandGesture_classification_CNN\my_variants\HG_capture.jpg"
    print(img_file_path, "img path")

    camera_instance.release()


    pass


#######################333
img_file_path = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_15_gujarati_char_recognition_CNN\Gui\build\assets\image_1.png"

window = tkinter.Tk()
# window.state("zoomed")
window.geometry("800x600")
window.title("Guj char recognizer")
# defining fonts
myFont = font.Font(family='Helvetica', size=20)


title_label = Label(window,text="GUJARATI CHARACTER RECOGNITION USING CNN", bg="#F7F8F6")
title_label['font'] = font.Font(family='Helvetica', size=20)
title_label.place(x=15, y=32, width=800, height=40)

right_side_objects = []

test_image = Label(window, image="", bg="#C4C4C4")
test_image.place(
    x=70,
    y=130,
    width=450,
    height=450
)

# right side panel

capture_btn = Button(window, text="Capture", bg="#429ef5", command=on_Capture)
right_side_objects.append(capture_btn)

brows_btn = Button(window, text="brows", bg="#429ef5", command=on_Brows)
right_side_objects.append(brows_btn)


classify_btn = Button(window, text="classify", bg="#429ef5", command=on_Classify)
right_side_objects.append(classify_btn)


prediction_label = Label(window, text="prediction", bg="#FCFF5F")
prediction_label['font'] = myFont
right_side_objects.append(prediction_label)


y = 210
for object in right_side_objects:
    if object == prediction_label:
        object.place(
            x=550,
            y=y,
            width=200,
            height=40
        )
    else:
        object.place(
            x=600,
            y=y,
            width=100,
            height=40
        )
    y += 80

# train_btn = Button(window, text="train", bg="#429ef5", command=on_Train)
# train_btn.place(
#     x=600,
#     y=370,
#     width=100,
#     height=40
# )

window.mainloop()