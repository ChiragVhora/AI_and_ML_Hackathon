import cv2

camera_instance = cv2.VideoCapture(0)   # getting camera instance for image capture, 0- inbuilt/primary, 1- secondary/external ...
x=0
while True:
    img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # img_from_cam = camera_instance.read()[1]         : only for getting img not flag

    if img_flag:    # if got the image then show
        window_title = str("Video frame by frame")
        cv2.imshow(window_title, img_from_cam)    # for window to open with title specified

        '''
        # uncomment to understand working of cv2.absdiff() --> for moving object detection 
        second_img = camera_instance.read()[1]
        img_diff = cv2.absdiff(img_from_cam,second_img)
        cv2.imshow("diff _ img", img_diff)
        '''

    else:
        print(f"problem getting image at x = {x}")

    # key = cv2.waitKey(50) & 0xFF  : for hexadecimal values ( basically key bitwise and-ed with 0xFF -> hex val)
    key = cv2.waitKey(50)    # cv2.waitKey(0)  : here 0 for still image until key press --> i.e. press key =>change image
    print(f"key pressed : {key} at x: {x}")
    x+=1

    if key == 27 or x>1000:     # 27 == Esc key , 32 == space key
        break

camera_instance.release()       # releasing camera instance ( releasing instances is good practice )
# realising the camera instance is must .
cv2.destroyAllWindows()     # destroying any window if left opened
print("done !")