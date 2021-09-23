import os

from keras.models import  load_model
import numpy as np
from PIL import Image

# variables #########################
RD_model_name = "Road_sign_recognition_model.h5"
classes = []
test_path = r""


# classify
def classify(img_path, cnn_model):
    target_size = (128, 128, 1)

    global success_count
    global classes
    # this converts img to 256x256 gray image
    test_image = Image.open(test_path)

    res_img = test_image.resize((30, 30))

    test_image_arr_expanded = np.expand_dims(res_img, axis=0)

    test_image_arr_expanded_to_np_arr = np.array(test_image_arr_expanded)

    results = cnn_model.predict(test_image_arr_expanded_to_np_arr)    # passing image arr as we train for that

    # print("result : ", results)
    pred_arr = np.argmax(results[0])
    # print("arr : ", arr)

    # maxx = np.amax(arr)
    max_prob = pred_arr.argmax(axis=0)
    # max_prob += 1
    prediction = classes[max_prob]
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

    return prediction