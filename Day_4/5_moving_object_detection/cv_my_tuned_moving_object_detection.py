import cv2
import imutils
import time

camera_instance = cv2.VideoCapture(
    0)  # getting camera instance for image capture, 0- inbuilt/primary, 1- secondary/external ...
x = 0

First_frame = None  # for comparison to know which part of img is changing
area = 500  # for small computational overhead (check : take original and check computational time )


def appy_rectengle_to_img_using_contours(img, outlines_cnts_from_imutils):
    text = "None"
    for outline_ctn_arr in outlines_cnts_from_imutils:  # for all moving object coordinate detected
        if cv2.contourArea(outline_ctn_arr) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(outline_ctn_arr)    # gives location for rectangle from single contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # add rect. to img
        text = "Moving Object Detected"
    # print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)     # add text to fixed loc
    pass


while True:
    img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # img_from_cam = camera_instance.read()[1]         : only for getting img not flag
    text = "Normal"


    if img_flag:
        ''' 
            if logic :                          ( it's not much don't panic :  
                                                    go through steps and check reference if needed)
                if img obtained -> Steps:
                    1. get resized img (small for computational overhead)
                    # get gray img of the same ( this approach works best for B/W img (binary type) )  
                    2. get gray img -> apply gaussian blur( for sharpness(noise) reduction )
                    3. set first frame for comparison
                    4. get img difference of current frame vs Old/(First_frame) : (for checking which part of img changed)
                    5. get threshold(B/W binary img) of difference img -> gives nice B/W object outlines 
                    6. refine threshold img by dilate and erode operation : ( for more accuracy )
                    7. get contours/ outlines from that refined img
                    8. we got locations , now draw rectangle 
                    9. show img, hurray ! 
                    
                    10. release resources 
                    
        '''
        ''' 
                   reference video for noise reducing : https://youtu.be/anBCL5Tta1g
                   reference video for contours : https://youtu.be/FbR9Xr0TVdY
                   continue : https://stackoverflow.com/questions/27035672/cv-extract-differences-between-two-images

        '''
        # step 1
        resized_original_img = imutils.resize(img_from_cam, width=area)  # 500

        # step 2
        resized_gray_img = cv2.cvtColor(resized_original_img, cv2.COLOR_BGR2GRAY)  # for converting B/W ( binary img)
        resized_gaussian_blurred_img = cv2.GaussianBlur(resized_gray_img, (9, 9), 0)

        # step 3
        if First_frame is None:
            First_frame = resized_gaussian_blurred_img
            continue
        # last is first frame
        if x % 3 == 0:  # comparing to every 3rd new frame
            First_frame = resized_gaussian_blurred_img

        # step 4
        img_difference = cv2.absdiff(First_frame, resized_gaussian_blurred_img)

        # step 5
        check_tuning_threshhold_pixel_val = 10
        ret, thresholded_BW_img = cv2.threshold(img_difference, check_tuning_threshhold_pixel_val, 255, cv2.THRESH_BINARY)  # [1]

        # step 6
        check_tuning_times_operation_perform = 20
        noise_removed_thresholded_BW_img = cv2.dilate(thresholded_BW_img, None,
                                                      iterations=check_tuning_times_operation_perform)  # append white pixels at boundry
        check_tuning_times_operation_perform = 3
        noise_removed_thresholded_BW_img_erode = cv2.erode(noise_removed_thresholded_BW_img, None,
                                                           iterations=check_tuning_times_operation_perform)  # removing extra white pixels

        # step 7
        outlines_cnts_from_cv2 = cv2.findContours(noise_removed_thresholded_BW_img_erode.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierachy = cv2.findContours(noise_removed_thresholded_BW_img_erode.copy(), cv2.RETR_EXTERNAL,
        #                                       cv2.CHAIN_APPROX_SIMPLE)
        outlines_cnts_from_imutils = imutils.grab_contours(outlines_cnts_from_cv2)

        # not working cv2.drawContours(resized_original_img, outlines_cnts_from_cv2, -1, (0,255,0), 3)
        # cv2.drawContours(resized_original_img, contours,-1, (0,255,0), 2)     # for exact boundry it get : noisy

        # step 8
        # ----- my func
        appy_rectengle_to_img_using_contours(resized_original_img, outlines_cnts_from_imutils)

        # no need--
        # appy_rectengle_to_img_using_contours(resized_gray_img, outlines_cnts_from_imutils)
        # appy_rectengle_to_img_using_contours(resized_gaussian_blurred_img, outlines_cnts_from_imutils)
        # appy_rectengle_to_img_using_contours(img_difference, outlines_cnts_from_imutils)
        # appy_rectengle_to_img_using_contours(thresholded_BW_img, outlines_cnts_from_imutils)
        # appy_rectengle_to_img_using_contours(noise_removed_thresholded_BW_img, outlines_cnts_from_imutils)

        # step 9
        cv2.imshow("Object Detector original", resized_original_img)
        cv2.imshow("gray detector", resized_gray_img)
        cv2.imshow("resized_gaussian_blurred_img detector", resized_gaussian_blurred_img)
        cv2.imshow("img_difference detector", img_difference)
        cv2.imshow("thresholded_BW_img detector", thresholded_BW_img)
        cv2.imshow("noise_removed_thresholded_BW_img detector", noise_removed_thresholded_BW_img)


    else:
        print(f"problem getting image at x = {x}")

    # key = cv2.waitKey(50) & 0xFF  : for hexadecimal values ( basically key bitwise and-ed with 0xFF -> hex val)
    key = cv2.waitKey(2)  # cv2.waitKey(0)  : here 0 for still image until key press --> i.e. press key =>change image
    # print(f"key pressed : {key} at x: {x}")
    x += 1

    if key == 27:  # or x>1000:     # 27 == Esc key , 32 == space key
        break

# step 10
cv2.destroyAllWindows()  # destroying any window if left opened
camera_instance.release()  # releasing camera instance ( releasing instances is good practice )
# realising the camera instance is must .
print("done !")