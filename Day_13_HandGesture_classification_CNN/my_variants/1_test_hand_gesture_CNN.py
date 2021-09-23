from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
import os

# variables ################################
# classes = ['01_palm', '02_l', '03_fist', '04_fist_moved', "05_thumb", "06_index", "07_ok", "08_palm_moved", "09_c", "10_down"]
from common_python_files.save_OR_load_model_into_json import cmn_load_nn_model
from main import cmd_progressBar

classes = ["index finger", "ok gesture", "palm", "pinky finger", "thumb down", "thumb up"]


# functions ################################

success_count = 0
def classify(img_path):
    (kernel, input_shape) = ((2, 2), (100, 100, 1))
    target_size = (input_shape[0], input_shape[1])

    global success_count
    global classes
    global cnn_model
    # this converts img to 256x256 gray image
    test_image = image.load_img(img_path, target_size=target_size, grayscale=True) # for the target we trained

    test_image_arr = image.img_to_array(test_image)
    test_image_arr_expanded = np.expand_dims(test_image_arr, axis=0)

    results = cnn_model.predict(test_image_arr_expanded)    # passing image arr as we train for that

    # print("result : ", results)
    pred_val = np.argmax(results[0])
    # print("arr : ", arr)

    # maxx = np.amax(arr)
    # max_prob = arr.argmax(axis=0)
    # max_prob += 1
    prediction = classes[pred_val]
    # prediction = classes[max_prob-1]

    first_3_char = prediction[:3]      # i.e 01_palm -> 01
    classified_file_3_char = f"{first_3_char}"
    print(classified_file_3_char)

    image_name = os.path.basename(img_path)
    if classified_file_3_char in image_name:    # in img name
        success_count+=1
        print("correct : ", image_name, " identified as ", prediction)
    else:
        print("False : ", image_name, " identified as ", prediction)


# load model #############################333
model_name = "hand_gesture_cnn_model"
cnn_model = cmn_load_nn_model(model_name)
print("model loaded.")

# get all images from test folder ###############################
test_folder_path = r"/Day_13_HandGesture_classification_CNN/Handgesture_Dataset/test"
from common_python_files.get_image_path import cmn_get_all_image_path_from_folder
image_files_path = cmn_get_all_image_path_from_folder(test_folder_path, include_sub_folder=True, files_to_get_per_folder=10)

print(image_files_path)
total_imgs = len(image_files_path)

exc_loss =0
# testing
progress_count = 0
for img_path in image_files_path:

# try:
    classify(img_path)
# except Exception:
#     exc_loss +=1
#     print(f"exception for img path : {img_path}")

    progress_count+=1
    cmd_progressBar(progress_count, total_imgs, message="Identifying")

total_imgs -= exc_loss      # we are not counting the failures in total
# result
print(f"\n\naccuracy : {success_count}/{total_imgs} =  {success_count /total_imgs *100} ")
