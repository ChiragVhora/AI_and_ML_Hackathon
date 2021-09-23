from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np
import os

# variables ################################
from common_python_files.save_OR_load_model_into_json import cmn_load_nn_model
from main import cmd_progressBar

classes = [ "ALA", "ANA", "B", "BHA", "CH", "CHH", "D", "DA", "DH", "DHA", "F", "G", "GH", "GNA", "H", "J", "JH", "K", "KH", "KSH", "L", "M", "N", "P", "R", "S", "SH", "SHH", "T", "TA", "TH", "THA", "V", "Y"]

# functions ################################

success_count = 0
def classify_gujarati_classifier(img_path):
    model_name = "my_var_Gujarati_classifier_cnn_model"
    cnn_model = cmn_load_nn_model(model_name)
    print("model loaded.")
    pred = classify(img_path, cnn_model)
    return pred
    pass

def classify(img_path, cnn_model):
    target_size = (128, 128, 1)

    global success_count
    global classes
    # this converts img to 256x256 gray image
    test_image = image.load_img(img_path, target_size=target_size, color_mode="grayscale") # for the target we trained

    test_image_arr = image.img_to_array(test_image)
    test_image_arr_expanded = np.expand_dims(test_image_arr, axis=0)

    results = cnn_model.predict(test_image_arr_expanded)    # passing image arr as we train for that

    # print("result : ", results)
    prediction = np.argmax(results[0])
    # print("pred index : ", prediction)

    # maxx = np.amax(arr)
    # max_prob = pred_arr.argmax(axis=0)
    # max_prob += 1
    # print("max prob : ", max_prob)
    # prediction = classes[max_prob]
    # prediction = classes[max_prob-1]

    # first_3_char = prediction[:3]      # i.e 01_palm -> 01
    # classified_file_3_char = f"{first_3_char}"
    # print(classified_file_3_char)
    pred_class = classes[int(prediction)]
    image_name = os.path.basename(img_path)
    if pred_class in image_name:    # in img name
        success_count+=1
        print("correct : ", image_name, " identified as ", pred_class)
    else:
        print("False : ", image_name, " identified as ", pred_class)

    return prediction


def load_and_classify_from_folder():
    # load model #############################333

    model_name = "my_var_Gujarati_classifier_cnn_model"
    cnn_model = cmn_load_nn_model(model_name)
    print("model loaded.")

    # get all images from test folder ###############################

    test_folder_path =  r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_15_gujarati_char_recognition_CNN\Dataset\test"

    from common_python_files.get_image_path import cmn_get_all_image_path_from_folder
    image_files_path = cmn_get_all_image_path_from_folder(test_folder_path, include_sub_folder=True)

    print(image_files_path)
    total_imgs = len(image_files_path)

    exc_loss =0
    # testing
    progress_count = 0
    for img_path in image_files_path:

    # try:
        classify(img_path, cnn_model)
    # except Exception:
    #     exc_loss +=1
    #     print(f"exception for img path : {img_path}")

        progress_count+=1
        cmd_progressBar(progress_count, total_imgs, message="Identifying")

    total_imgs -= exc_loss      # we are not counting the failures in total
    # result
    print(f"\n\naccuracy : {success_count}/{total_imgs} =  {success_count /total_imgs *100} ")
    pass


load_and_classify_from_folder()