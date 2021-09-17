import numpy as np
from keras.preprocessing import image

# functions #################################3

success_count = 0
def classify(img_path):
    global success_count
    global cnn_model
    image_name = os.path.basename(img_path)

    test_image = image.load_img(img_path, target_size=(64, 64)) # for the target we trained

    test_image_arr = image.img_to_array(test_image)
    test_image_arr_expanded = np.expand_dims(test_image_arr, axis=0)

    results = cnn_model.predict(test_image_arr_expanded)    # passing image arr as we train for that

    prediction= ""
    if results[0][0] == 1:
        prediction = "thanos"
    else:
        prediction = "joker"

    if prediction in image_name:
        success_count+=1
        print("correct : ", image_name, " identified as ", prediction)
    else:
        print("False : ", image_name, " identified as ", prediction)


def get_model():
    # loading model
    from common_python_files.save_OR_load_model_into_jason import cmn_load_nn_model
    model_name = "cnn_nn_joker_and_thanos_recognition"

    cnn_model = cmn_load_nn_model(model_name)
    return cnn_model
    pass

##########################33

import os

cnn_model = get_model()

test_folder_path =  r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_12_Inage_classification_CNN\1_hackathon_Image_classification_using_cnn\Dataset\test"

from common_python_files.get_image_path import cmn_get_all_image_path_from_folder
image_files_path = cmn_get_all_image_path_from_folder(test_folder_path, include_sub_folder=True )

print(image_files_path)
total_imgs = len(image_files_path)

exc_loss =0
# testing
for img_path in image_files_path:
    try:
        classify(img_path)
    except Exception:
        exc_loss +=1
        print(f"exception for img path : {img_path}")

total_imgs -= exc_loss
# result
print(f"\n\naccuracy : {success_count}/{total_imgs} =  {success_count /total_imgs *100} ")