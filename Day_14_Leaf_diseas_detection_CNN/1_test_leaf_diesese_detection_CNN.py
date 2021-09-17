from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
import os

# variables ################################
from common_python_files.save_OR_load_model_into_jason import cmn_load_nn_model
from main import cmd_progressBar

classes = [ "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy", "Blueberry___healthy", "Cherry_(including_sour)___healthy", "Cherry_(including_sour)___Powdery_mildew", "Corn_(maize)___Common_rust_", "Corn_(maize)___healthy", "Corn_(maize)___Northern_Leaf_Blight", "Grape___Black_rot", "Grape___Esca_(Black_Measles)", "Grape___healthy", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy", "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight", "Potato___healthy", "Potato___Late_blight", "Raspberry___healthy", "Soybean___healthy", "Squash___Powdery_mildew", "Strawberry___healthy", "Strawberry___Leaf_scorch", "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy", "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot", "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus"]

# functions ################################

success_count = 0
def classify(img_path):
    target_size = (128, 128, 3)

    global success_count
    global classes
    global cnn_model
    # this converts img to 256x256 gray image
    test_image = image.load_img(img_path, target_size=target_size) # for the target we trained

    test_image_arr = image.img_to_array(test_image)
    test_image_arr_expanded = np.expand_dims(test_image_arr, axis=0)

    results = cnn_model.predict(test_image_arr_expanded)    # passing image arr as we train for that

    # print("result : ", results)
    pred_arr = np.argmax(results[0])
    # print("arr : ", arr)

    # maxx = np.amax(arr)
    # max_prob = pred_arr.argmax(axis=0)
    # max_prob += 1
    prediction = classes[pred_arr]
    # prediction = classes[max_prob-1]

    # first_3_char = prediction[:3]      # i.e 01_palm -> 01
    # classified_file_3_char = f"{first_3_char}"
    # print(classified_file_3_char)

    image_name = os.path.basename(img_path)
    if prediction in image_name:    # in img name
        success_count+=1
        print("correct : ", image_name, " identified as ", prediction)
    else:
        print("False : ", image_name, " identified as ", prediction)


# load model #############################333

model_name = "leaf_disease_cnn_model"
cnn_model = cmn_load_nn_model(model_name)
print("model loaded.")

# get all images from test folder ###############################

test_folder_path =  r"C:\Users\chira\Desktop\ML_ai_hackathon_aug_21\Day_14_leaf_diesese_detection\New Plant Diseases Dataset(Augmented)\New Plant Diseases Dataset(Augmented)\test"

from common_python_files.get_image_path import cmn_get_all_image_path_from_folder
image_files_path = cmn_get_all_image_path_from_folder(test_folder_path, include_sub_folder=True)

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
