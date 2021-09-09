import cv2
import imutils
import numpy

# open img file
img_src = cv2.imread("ironman_nanotech_suite.png")

# for image to fit in display (fit side by side comparison)
# resizing the image file, inter => interpolation method ( technique to reduce img)
resized_src_Img = imutils.resize(img_src, width=500, height=500, inter=cv2.INTER_AREA)

# changing color to gray of resized
gray_src_img = cv2.cvtColor(resized_src_Img, cv2.COLOR_BGR2GRAY)

# smoothing,  k-size should be odd
gaussianBlur_grayImg = cv2.GaussianBlur(gray_src_img, (5, 5), cv2.BORDER_DEFAULT)

# thresholding ( converting gray image into black and white image as-> if pixel_val > threshold then black else white)
thresholdImg = cv2.threshold(gaussianBlur_grayImg, 50, 255, cv2.THRESH_BINARY)[1]

# saving the resized file
cv2.imwrite("thresholdImg.png", thresholdImg)
cv2.imwrite("gaussian_gray_blurred_img.png", gaussianBlur_grayImg)
cv2.imwrite("resized_img.png", resized_src_Img)




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