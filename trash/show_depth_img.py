import cv2

# Read depth image
depth_img = cv2.imread('0.png', cv2.IMREAD_ANYDEPTH)

# show depth image
cv2.imshow('depth', depth_img)
cv2.waitKey(0)