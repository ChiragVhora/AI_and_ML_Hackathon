import cv2
import imutils

#################################################################

color_upper_hsv_val = (25, 255, 255)
color_lower_hsv_val = (10, 100, 20)

First_frame = None  # for comparison to know which part of img is changing
width = 500  # for small computational overhead (check : take original and check computational time )
x = 0

#################################################################


camera_instance = cv2.VideoCapture(
    0)  # getting camera instance for image capture, 0- inbuilt/primary, 1- secondary/external ...


def appy_rectengle_to_img_using_contours(img, outlines_cnts_from_imutils):
    text = "None"
    global width
    for outline_ctn_arr in outlines_cnts_from_imutils:  # for all moving object coordinate detected
        area = width
        if cv2.contourArea(outline_ctn_arr) < area:
            continue
        (x, y, w, h) = cv2.boundingRect(outline_ctn_arr)  # gives location for rectangle from single contour
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)  # add rect. to img
        text = "Moving Object Detected"
    # print(text)
    cv2.putText(img, text, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)  # add text to fixed loc
    pass


while True:
    img_flag, img_from_cam = camera_instance.read()  # read flag(read success Or Failure) and image from camera
    # img_from_cam = camera_instance.read()[1]         : only for getting img not flag

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
            reference 
                cv2.inRange : https://www.educba.com/opencv-inrange/
                finding centers of object : https://learnopencv.com/find-center-of-blob-centroid-using-opencv-cpp-python/
        '''

        # step 1
        resized_original_img = imutils.resize(img_from_cam, width=width)  # 500

        # step 2
        resized_gaussian_blurred_img = cv2.GaussianBlur(resized_original_img, (1, 1), 0)  # (1,1) -> intensity for small blur
        resized_HSV_img = cv2.cvtColor(resized_gaussian_blurred_img, cv2.COLOR_BGR2HSV)  # for converting to HSV img

        # step 3 - returned color ranged objects as white pixels
        only_color_range_objects_img = cv2.inRange(resized_HSV_img, color_lower_hsv_val, color_upper_hsv_val)

        # step 5
        # check_tuning_threshhold_pixel_val = 10
        # ret, thresholded_BW_img = cv2.threshold(img_difference, check_tuning_threshhold_pixel_val, 255,
        #                                         cv2.THRESH_BINARY)  # [1]

        # step 6
        check_tuning_times_operation_perform = 20
        noise_removed_objects_img_dilate = cv2.dilate(only_color_range_objects_img, None,
                                               iterations=check_tuning_times_operation_perform)  # append white pixels at boundry
        check_tuning_times_operation_perform = 3
        noise_removed_objects_img_erode = cv2.erode(noise_removed_objects_img_dilate, None,
                                                    iterations=check_tuning_times_operation_perform)  # removing extra white pixels

        # step 7
        outlines_cnts_from_img, hierarchy = cv2.findContours(noise_removed_objects_img_erode.copy(), cv2.RETR_EXTERNAL,
                                                  cv2.CHAIN_APPROX_SIMPLE)
        # contours, hierachy = cv2.findContours(noice_removed_thresholded_BW_img_erode.copy(), cv2.RETR_EXTERNAL,
        #                                       cv2.CHAIN_APPROX_SIMPLE)
        # outlines_cnts_from_imutils = imutils.grab_contours(outlines_cnts_from_cv2)

        center = None
        if len(outlines_cnts_from_img) > 0:     # we obtain color object then:
            max_contour_area = max(outlines_cnts_from_img, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(max_contour_area)

            M = cv2.moments(max_contour_area)
            Cx = int(M["m10"]/M["m00"])
            Cy = int(M["m01"]/M["m00"])
            center =(Cx, Cy)

            if radius>10:   # make sure obj is big enough

                color = (0, 255, 255)   # yellow
                cv2.circle(resized_original_img, (int(x), int(y)), int(radius), color, 1)
                color = (0, 0, 255)     # red
                cv2.circle(resized_original_img, center, 3, color, -1)

                print(f"small-center : {center}, radius : {radius}, big-(x,y) : ({x},{y})")
                if radius > width/2: # close to camera as width is only 500
                    print("stop : too close")
                else:
                    if Cx < 150: #width/3:
                        print("left")
                    elif Cx < 250:#width*2/3:
                        print("front")
                    elif Cx > 450:
                        print("right")
                    else:
                        print("stop")

        # step 9
        cv2.imshow("Object Detector original resized", resized_original_img)
        cv2.imshow("resized_gaussian_blurred_img detector", resized_gaussian_blurred_img)
        cv2.imshow("HSV detector", resized_HSV_img)
        cv2.imshow("only_color_range_objects_img detector", only_color_range_objects_img)
        cv2.imshow("noise_removed_objects_img_dilate detector", noise_removed_objects_img_dilate)
        cv2.imshow("noise_removed_objects_img_erode detector", noise_removed_objects_img_erode)


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
