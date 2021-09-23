# HSVtunerHere
import time

import cv2
import imutils

#################################################################
# this setting is for skin color ( as per my cam)
color_upper_hsv_val = (25, 255, 255)  # As (H, S, V)
color_lower_hsv_val = (10, 100, 20)

frame_width = 500  # for small computational overhead (check : take original and check computational time )
frame_height = 500
x = 0


############################ Trackers #####################################

def empty(a):
    pass


# CREATE TRACKBAR
cv2.namedWindow("Live HSV object color tuner")
# cv2.namedWindow("Live HSV object color tuner", cv2.WINDOW_AUTOSIZE)
cv2.resizeWindow("Live HSV object color tuner", 350, 450)
cv2.createTrackbar("HUE Low", "Live HSV object color tuner", 0, 255, empty)
cv2.createTrackbar("HUE High", "Live HSV object color tuner", 255, 255, empty)
cv2.createTrackbar("Saturation Low", "Live HSV object color tuner", 0, 255, empty)
cv2.createTrackbar("Saturation High", "Live HSV object color tuner", 255, 255, empty)
cv2.createTrackbar("Value Low", "Live HSV object color tuner", 0, 255, empty)
cv2.createTrackbar("Value High", "Live HSV object color tuner", 255, 255, empty)
cv2.createTrackbar("Brightness", "Live HSV object color tuner", 180, 255, empty)
cv2.createTrackbar("Contrast", "Live HSV object color tuner", 180, 255, empty)

cv2.createTrackbar("erode", "Live HSV object color tuner", 3, 50, empty)
cv2.createTrackbar("dilate", "Live HSV object color tuner", 10, 50, empty)

cv2.createTrackbar("object min area radius", "Live HSV object color tuner", 10, int(500/3 - 1), empty)

# hint to save settings from cv2
# cv2.createTrackbar("save settings ? 0-no , 1 - yes", "Live HSV object color tuner", 0, 1, save_settings)
#################################################################

camera_instance = cv2.VideoCapture(
    0)  # getting camera instance for image capture, 0- inbuilt/primary, 1- secondary/external ...


while True:
    img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # img_from_cam = camera_instance.read()[1]         : only for getting img not flag
    cameraBrightness = cv2.getTrackbarPos("Brightness", "Live HSV object color tuner")
    cameraContrast = cv2.getTrackbarPos("Contrast", "Live HSV object color tuner")
    camera_instance.set(10, cameraBrightness)
    camera_instance.set(cv2.CAP_PROP_CONTRAST, cameraContrast)

    H_low = cv2.getTrackbarPos("HUE Low", "Live HSV object color tuner")
    S_low = cv2.getTrackbarPos("Saturation Low", "Live HSV object color tuner")
    V_low = cv2.getTrackbarPos("Value Low", "Live HSV object color tuner")
    H_high = cv2.getTrackbarPos("HUE High", "Live HSV object color tuner")
    S_high = cv2.getTrackbarPos("Saturation High", "Live HSV object color tuner")
    V_high = cv2.getTrackbarPos("Value High", "Live HSV object color tuner")

    color_upper_hsv_val = (H_high, S_high, V_high)
    color_lower_hsv_val = (H_low, S_low, V_low)

    erode_tuning = cv2.getTrackbarPos("erode", "Live HSV object color tuner")
    dilate_tuning = cv2.getTrackbarPos("dilate", "Live HSV object color tuner")

    object_radius = cv2.getTrackbarPos("object min area radius", "Live HSV object color tuner")

    if img_flag:
        ''' 
            if logic :                          ( it's not much don't panic :  
                                                    go through steps and check reference if needed)
                if img obtained -> Steps:
                    1. get resized img (small for computational overhead)
                    # get gray img of the same ( this approach works best for B/W img (binary type) )  
                    2. apply gaussian blur( for sharpness(noise) reduction ) 
                       and get HSV img
                    3. get the color object img ( using inRange() fun)
                    4. refine object img by dilate and erode operation : ( for more accuracy )
                    5. get contours/ outlines from that refined img
                    6. get locations of center, rectangle , radius 
                    7. we got locations , now draw circle and rectangle 
                    8. text field for object location in img , using 3X3 parts logic
                    9. show img, hurray ! 
                    10. release resources 

        '''
        '''
            reference 
                cv2.inRange : https://www.educba.com/opencv-inrange/
                finding centers of object : https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        '''

        # step 1
        resized_original_img = imutils.resize(img_from_cam, width=frame_width, height=frame_height)  # 500

        # step 2
        resized_gaussian_blurred_img = cv2.GaussianBlur(resized_original_img, (1, 1),
                                                        0)  # (1,1) -> intensity for small blur
        resized_HSV_img = cv2.cvtColor(resized_gaussian_blurred_img, cv2.COLOR_BGR2HSV)  # for converting to HSV img

        # step 3 - returned color ranged objects as white pixels
        only_color_range_objects_img = cv2.inRange(resized_HSV_img, color_lower_hsv_val, color_upper_hsv_val)

        # step 4
        noise_removed_objects_img_dilate = cv2.dilate(only_color_range_objects_img, None,
                                                      iterations=dilate_tuning)  # append white pixels at boundry
        noise_removed_objects_img_erode = cv2.erode(noise_removed_objects_img_dilate, None,
                                                    iterations=erode_tuning)  # removing extra white pixels

        # step 5
        outlines_cnts_from_img, hierarchy = cv2.findContours(noise_removed_objects_img_erode.copy(), cv2.RETR_EXTERNAL,
                                                             cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierachy = cv2.findContours(noice_removed_thresholded_BW_img_erode.copy(), cv2.RETR_EXTERNAL,
        #                                       cv2.CHAIN_APPROX_SIMPLE)
        # outlines_cnts_from_imutils = imutils.grab_contours(outlines_cnts_from_cv2)

        center = None
        if len(outlines_cnts_from_img) > 0:  # we obtain color object then:
            # step 6
            max_contour_area = max(outlines_cnts_from_img, key=cv2.contourArea)
                # for circle
            ((C_x, C_y), radius) = cv2.minEnclosingCircle(max_contour_area)
            center = (int(C_x), int(C_y))
                # for rectangle
            (x, y, w, h) = cv2.boundingRect(max_contour_area)  # gives location for rectangle from single contour

            # no need to calculate center again
            # M = cv2.moments(max_contour_area)     # moments is weighted sum
            # Cx = int(M["m10"]/M["m00"])
            # Cy = int(M["m01"]/M["m00"])
            # center =(Cx, Cy)

            if radius > object_radius:  # make sure obj is big enough, small-> skip
                # step 7
                color = (0, 255, 255)  # yellow
                # cv2.circle(resized_original_img, (int(x), int(y)), int(radius), color, 1)
                cv2.circle(resized_original_img, center, int(radius), color, 1)
                color = (0, 0, 255)  # red
                cv2.circle(resized_original_img, center, 3, color, -1)  # -1 fills the circle with color

                cv2.rectangle(resized_original_img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # add rect. to img

                # print(f"small-center : {center}, radius : {radius}, big-(x,y) : ({C_x},{C_y})")
                # print(f"center vs frame : {center} vs (500,500)")

                # step 8
                frame_height,frame_width,_ = resized_original_img.shape     # height going to be different than specified before 
                object_location_text = ""
                if radius > frame_width / 3:  # close to camera as frame_width is only 500
                    object_location_text = "Too close (stop)"
                else:
                    # divided into 3X3 parts ,by simple division
                    # check center location for which part it belongs to

                    # Top 3 parts
                    if C_x < frame_width / 3 and C_y < frame_height / 3:
                        object_location_text = "top left"
                    elif C_x < frame_width * 2 / 3 and C_y < frame_height / 3:
                        object_location_text = "top center"
                    elif C_x > frame_width * 2 / 3 and C_y < frame_height / 3:
                        object_location_text = "top right"

                    # middle 3 parts
                    elif C_x < frame_width / 3 and C_y < frame_height * 2 / 3:
                        object_location_text = "center left"
                    elif C_x < frame_width * 2 / 3 and C_y < frame_height * 2 / 3:
                        object_location_text = "center"
                    elif C_x > frame_width * 2 / 3 and C_y < frame_height * 2 / 3:
                        object_location_text = "center right"
                    # bottom 3 parts
                    elif C_x < frame_width / 3 and C_y > frame_height * 2 / 3:
                        object_location_text = "bottom left"
                    elif C_x < frame_width * 2 / 3 and C_y > frame_height * 2 / 3:
                        object_location_text = "bottom center"
                    elif C_x > frame_width * 2 / 3 and C_y > frame_height * 2 / 3:
                        object_location_text = "bottom right"
                    else:
                        object_location_text = "move the object"
            else:
                object_location_text = "object too small !"
        else:
            object_location_text = "not detected"


        object_location_text = "Object Location : " + object_location_text
        print(object_location_text)
        cv2.putText(resized_original_img, object_location_text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)  # add text to fixed loc

        # step 9
        cv2.imshow("Color Object Detector", resized_original_img)
        # cv2.imshow("resized_gaussian_blurred_img detector", resized_gaussian_blurred_img)
        # cv2.imshow("HSV detector", resized_HSV_img)
        # cv2.imshow("only_color_range_objects_img detector", only_color_range_objects_img)
        cv2.imshow("Object Detected by tuner", noise_removed_objects_img_dilate)
        # cv2.imshow("noise_removed_objects_img_erode detector", noise_removed_objects_img_erode)

        # cv2.imshow("Live HSV object color tuner", noise_removed_objects_img_erode)
        # import numpy as np
        # Hori = np.concatenate((noise_removed_objects_img_erode, resized_original_img), axis=1)
        # cv2.imshow("Object Tracker", Hori)

        # vertical = np.concatenate((noise_removed_objects_img_erode), axis=0)

        # cv2.imshow("Live HSV object color tuner", resized_original_img)

        time.sleep(0.1)
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
