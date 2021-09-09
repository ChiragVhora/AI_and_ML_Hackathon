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

    # check_numpy_list_comprehension()
    check_fisher_face_recognizer()