import cv2
import imutils
import numpy

# open img file
img_src = cv2.imread("ironman_nanotech_suite.png")

# for image to fit in display (fit side by side comparison)
# resizing the image file, inter => interpolation method ( technique to reduce img)
resized_src_Img = imutils.resize(img_src, width=500, height=500, inter=cv2.INTER_AREA)


# k-size should be odd
gaussianBlurImg = cv2.GaussianBlur(resized_src_Img, (21, 21), cv2.BORDER_DEFAULT)

# saving the resized file
cv2.imwrite("gaussianBlurImg.png", gaussianBlurImg)




# extra
# display input and output image
cv2.imshow("Gaussian Smoothing",numpy.hstack((resized_src_Img, gaussianBlurImg)))
cv2.waitKey(0) # waits until a key is pressed
cv2.destroyAllWindows() # destroys the window showing image
# Done
print("Done !")

'''
    reference : https://www.tutorialkart.com/opencv/python/opencv-python-gaussian-image-smoothing/ 
'''