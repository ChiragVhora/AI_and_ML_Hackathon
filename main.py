# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
def check_numpy_list_comprehension():
    import numpy
    images = (1,2,3,4,5,6)
    labels = [1,1,1,0,0,0]
    (images, labels) = [numpy.array(lis) for lis in [images, labels]]
    print(f"images type {type(images)} data : {images} , labels - type{type(labels)} data : {labels}")
    pass


def check_fisher_face_recognizer():
    import cv2
    model = cv2.face.FisherFaceRecognizer_create()
    print(model)
    pass


def make_list_from_list_of_tuples_for_prediction():
    prediction = [(x/10, 1-x/10) for x in range(10)]
    # print(prediction)
    true_pred_list = [true_pred for _,true_pred in prediction]
    print(true_pred_list)
    pass


def check_negative_slicing():
    list_ = [i for i in range(10)]
    slice_list = list_[:-2]
    print(f"list : {list_} \nSliced : {slice_list}")
    pass


def check_pandas():
    import pandas as pd
    # import kerastuner
    df = pd.DataFrame({'X': [78, 85, 96, 80, 86], 'Y': [84, 94, 89, 83, 86], 'Z': [86, 97, 96, 72, 83]});
    print(df)
    pass


if __name__ == '__main__':
    # print_hi('PyCharm')
    # import cv2
    # import imutils
    #
    # frame_width = 600
    # frame_height = 600
    #
    # camera_instance = cv2.VideoCapture(0)
    # img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # resized_original_img = imutils.resize(img_from_cam, width=frame_width, height=frame_height)  # 500
    # shape = resized_original_img.shape
    # frame_height, frame_width, _ = resized_original_img.shape
    # print(frame_height, frame_width, _)

    # function prob tried
    # check_numpy_list_comprehension()
    # check_fisher_face_recognizer()
    # make_list_from_list_of_tuples_for_prediction()
    # check_negative_slicing()
    check_pandas()