import cv2
import imutils

# open img file
img = cv2.imread("ironman_nanotech_suite.png")

# resizing the image file, inter => interpolation method ( technique to reduce img)
resizedImg = imutils.resize(img, width=500, height=500, inter=cv2.INTER_AREA)

# saving the resized file
cv2.imwrite("resizedImg.png", resizedImg)
# Done
print("Done !")

'''
    reference : https://www.pyimagesearch.com/2021/01/20/opencv-resize-image-cv2-resize/ 
'''