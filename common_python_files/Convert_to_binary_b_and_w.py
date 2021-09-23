import cv2
import imutils
import os
import numpy

folder = r"C:\Users\chira\PycharmProjects\AI_and_ML_Hackathon\Day_13_HandGesture_classification\Handgesture_Dataset"
# cwd = os.getcwd()
folder = r"C:\Users\chira\Desktop\ML_ai_hackathon_aug_21\Handgesture_Dataset\test\index finger"
img_ext = ("jpg", "jpeg", "png", "tiff")
color_upper_hsv_val = (25, 255, 255)
color_lower_hsv_val = (10, 100, 20)

def change_image_to_binary(folder):
    for root, Dirs, files in os.walk(folder):
        i = 0
        # folder_name = os.path.basename(root)
        # print(folder_name)

        for old_file in files:
            i += 1
            old_file_ext = old_file.split(".")[-1]
            print(old_file_ext)
            if old_file_ext in img_ext:
                old_path = os.path.join(root, old_file)
                # open img file
                print(old_path)
                img_src = cv2.imread(old_path)

                # print(img_src)
                # time.sleep()

                HSV_image = cv2.cvtColor(img_src, cv2.COLOR_BGR2HSV)  # for converting to HSV img

                # step 3 - returned color ranged objects as white pixels
                only_color_range_objects_img = cv2.inRange(HSV_image, color_lower_hsv_val, color_upper_hsv_val)

                check_tuning_times_operation_perform = 3
                noise_removed_objects_img_dilate = cv2.dilate(only_color_range_objects_img, None,
                                                              iterations=check_tuning_times_operation_perform)  # append white pixels at boundry
                check_tuning_times_operation_perform = 1
                noise_removed_objects_img_erode = cv2.erode(noise_removed_objects_img_dilate, None,
                                                            iterations=check_tuning_times_operation_perform)  # removing extra white pixels

                # for image to fit in display (fit side by side comparison)
                # resizing the image file, inter => interpolation method ( technique to reduce img)
                # resized_src_Img = imutils.resize(img_src, width=500, height=500, inter=cv2.INTER_AREA)

                # changing color to gray of resized
                # gray_src_img = cv2.cvtColor(only_color_range_objects_img, cv2.THRESH_BINARY)

                # smoothing,  k-size should be odd
                # gaussianBlur_grayImg = cv2.GaussianBlur(gray_src_img, (5, 5), cv2.BORDER_DEFAULT)

                # thresholding ( converting gray image into black and white image as-> if pixel_val > threshold then black else white)
                # thresholdImg = cv2.threshold(gray_src_img, 50, 255, cv2.THRESH_BINARY)[1]

                # saving the resized file
                new_name = "binary_" + str(old_file)
                # os.remove(old_path)
                # new_binary_path = os.path.join(root ,new_name)
                # cv2.imwrite(new_binary_path, noise_removed_objects_img_erode)
                # print(new_name)

                # show
                cv2.imshow("img_src to ", img_src)
                cv2.imshow("HSV_image to ", HSV_image)
                cv2.imshow("only_color_range_objects_img to ", only_color_range_objects_img)
                # cv2.imshow("gray_src_img to ", gray_src_img)
                cv2.imshow("thresholdImg to ", noise_removed_objects_img_erode)

                cv2.waitKey(0)



# if __name__ == "Convert_to_binary_b_and_w":

change_image_to_binary(folder)




# extra
# display input and output image
# cv2.imshow("Gaussian Smoothing", numpy.hstack((resized_src_Img, gaussianBlur_grayImg, thresholdImg)))
# cv2.waitKey(0) # waits until a key is pressed
# cv2.destroyAllWindows() # destroys the window showing image
# Done
print("Done !")

'''
    reference : https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/ 
'''