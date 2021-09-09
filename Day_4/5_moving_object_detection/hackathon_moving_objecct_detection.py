import cv2
import imutils
import time

camera_instance = cv2.VideoCapture(0)   # getting camera instance for image capture, 0- inbuilt/primary, 1- secondary/external ...
x=0

First_frame = None
area = 500


def appy_rectengle_to_img_using_contours(img, outlines_cnts_from_imutils):
    text = "None"
    for outline_ctn_arr in outlines_cnts_from_imutils:
        if cv2.contourArea(outline_ctn_arr) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(outline_ctn_arr)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text = "Moving Object Detected"
    # print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    pass


while True:
    img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # img_from_cam = camera_instance.read()[1]         : only for getting img not flag
    text = "Normal"

    if img_flag:    # if got the image then show
        resized_original_img = imutils.resize(img_from_cam, width=area)     # 500
        resized_gray_img = cv2.cvtColor(resized_original_img, cv2.COLOR_BGR2GRAY)   # for converting B/W ( binary img)
        resized_gaussian_blurred_img = cv2.GaussianBlur(resized_gray_img, (5,5), 0)
        if First_frame is None:
            First_frame = resized_gaussian_blurred_img
            continue

        img_difference = cv2.absdiff(First_frame, resized_gaussian_blurred_img)
        ret,thresholded_BW_img = cv2.threshold(img_difference, 10, 255, cv2.THRESH_BINARY)     #[1]
        noice_removed_thresholded_BW_img = cv2.dilate(thresholded_BW_img, None, iterations=3)   # append white pixels at boundry
        ''' 
            reference video for noise reducing : https://youtu.be/anBCL5Tta1g
            reference video for contours : https://youtu.be/FbR9Xr0TVdY
            continue : https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images
            
        '''
        outlines_cnts_from_cv2 = cv2.findContours(noice_removed_thresholded_BW_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours,hierachy = cv2.findContours(noice_removed_thresholded_BW_img.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        outlines_cnts_from_imutils = imutils.grab_contours(outlines_cnts_from_cv2)

        # -- created function to add rectangles to image from contours
        # appy_rectengle_to_img_using_contours(resized_original_img, outlines_cnts_from_imutils)
        appy_rectengle_to_img_using_contours(resized_original_img, contours)

        cv2.imshow("Object Detector original", resized_original_img)
        # cv2.imshow("gray detector", resized_gray_img)
        # cv2.imshow("resized_gaussian_blurred_img detector", resized_gaussian_blurred_img)
        # cv2.imshow("img_difference detector", img_difference)
        # cv2.imshow("thresholded_BW_img detector", thresholded_BW_img)
        # cv2.imshow("noice_removed_thresholded_BW_img detector", noice_removed_thresholded_BW_img)


    else:
        print(f"problem getting image at x = {x}")

    # key = cv2.waitKey(50) & 0xFF  : for hexadecimal values ( basically key bitwise and-ed with 0xFF -> hex val)
    key = cv2.waitKey(2)    # cv2.waitKey(0)  : here 0 for still image until key press --> i.e. press key =>change image
    # print(f"key pressed : {key} at x: {x}")
    x+=1

    if key == 27 : #or x>1000:     # 27 == Esc key , 32 == space key
        break


cv2.destroyAllWindows()     # destroying any window if left opened
camera_instance.release()       # releasing camera instance ( releasing instances is good practice )
# realising the camera instance is must .
print("done !")