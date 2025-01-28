"""This script is used to detect the mire on the capillaroscopies"""

import cv2
import matplotlib.pyplot as plt 

# read the image
# image = cv2.imread('nat.png')


# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# cv2.imshow('Gray', img_gray)
# cv2.waitKey(0)

# # detect contours and without thresholding
# contours, hierarchy = cv2.findContours(image=img_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

# # Filter out external contours
# # external_contours = []
# # for i in range(len(contours)):
# #     if hierarchy[0][i][3] == -1:  # Check if contour has no parent (external contour)
# #         external_contours.append(contours[i])

# # draw contours on the original image
# image_contour = image.copy()
# cv2.drawContours(image=image_contour, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=2, lineType=cv2.LINE_AA)
# # see the results
# cv2.imshow('Contour detection using gray images', image_contour)
# # print(f"CCOMP: {hierarchy}")
# cv2.waitKey(0)

from skimage.measure import regionprops

# img_path = "ensembles/masks/N22amask.jpg"

# img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

# image = cv2.imread(img_path)

# contours, hierarchy = cv2.findContours(img,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

# contour_area = cv2.contourArea(contours[0])
# cont_perimeter = cv2.arcLength(contours[0], True)

# print(contour_area)

# cv2.drawContours(image=image, contours=contours, contourIdx=-1, color=(255, 255, 255), thickness=1, lineType=cv2.LINE_AA)
# # see the results
# cv2.imshow('Contour detection using gray images', image)
# # # print(f"CCOMP: {hierarchy}")
# cv2.waitKey(0)

# print(len(contours))


image = cv2.imread('ensembles/masks/N43amask.jpg') 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
blur = cv2.GaussianBlur(gray, (11, 11), 0) 
plt.imshow(blur, cmap='gray') 

canny = cv2.Canny(blur, 30, 150, 3) 
plt.imshow(canny, cmap='gray') 


dilated = cv2.dilate(canny, (1, 1), iterations=2) 
plt.imshow(dilated, cmap='gray') 


(cnt, hierarchy) = cv2.findContours( 
    dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2) 
  
plt.imshow(rgb) 

print("capillars in the image : ", len(cnt)) 

plt.show()





	
# # convert the image to grayscale format
# img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# # apply binary thresholding
# ret, thresh = cv2.threshold(img_gray, 150, 255, cv2.THRESH_BINARY)
# # visualize the binary image
# cv2.imshow('Binary image', thresh)
# cv2.waitKey(0)
# cv2.imwrite('image_thres1.jpg', thresh)
# cv2.destroyAllWindows()

# # detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
# contours, hierarchy = cv2.findContours(image=thresh, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
                                      
# # draw contours on the original image
# image_copy = image.copy()
# cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
                
# # see the results
# cv2.imshow('None approximation', image_copy)
# cv2.waitKey(0)
# cv2.imwrite('contours_none_image1.jpg', image_copy)
# cv2.destroyAllWindows()



# image_copy2 = image.copy()
# cv2.drawContours(image_copy2, contours, -1, (0, 255, 0), 2, cv2.LINE_AA)
# cv2.imshow('SIMPLE Approximation contours', image_copy2)
# cv2.waitKey(0)
# image_copy3 = image.copy()
# for i, contour in enumerate(contours): # loop over one contour area
#    for j, contour_point in enumerate(contour): # loop over the points
#        # draw a circle on the current contour coordinate
#        cv2.circle(image_copy3, ((contour_point[0][0], contour_point[0][1])), 2, (0, 255, 0), 2, cv2.LINE_AA)
# # see the results
# cv2.imshow('CHAIN_APPROX_SIMPLE Point only', image_copy3)
# cv2.waitKey(0)
 
# # detect contours using blue channel and without thresholding
# contours1, hierarchy1 = cv2.findContours(image=img_gray, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
 
 
## B, G, R channel splitting
# blue, green, red = cv2.split(image)
 
# # draw contours on the original image
# image_contour_blue = image.copy()
# cv2.drawContours(image=image_contour_blue, contours=contours1, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# # see the results
# cv2.imshow('Contour detection using blue channels only', image_contour_blue)
# cv2.waitKey(0)
# cv2.imwrite('blue_channel.jpg', image_contour_blue)
# cv2.destroyAllWindows()
 
# # detect contours using green channel and without thresholding
# contours2, hierarchy2 = cv2.findContours(image=green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# # draw contours on the original image
# image_contour_green = image.copy()
# cv2.drawContours(image=image_contour_green, contours=contours2, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# # see the results
# cv2.imshow('Contour detection using green channels only', image_contour_green)
# cv2.waitKey(0)
# cv2.imwrite('green_channel.jpg', image_contour_green)
# cv2.destroyAllWindows()
 
# # detect contours using red channel and without thresholding
# contours3, hierarchy3 = cv2.findContours(image=red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
# # draw contours on the original image
# image_contour_red = image.copy()
# cv2.drawContours(image=image_contour_red, contours=contours3, contourIdx=-1, color=(0, 255, 0), thickness=2, lineType=cv2.LINE_AA)
# # see the results
# cv2.imshow('Contour detection using red channels only', image_contour_red)
# cv2.waitKey(0)
# cv2.imwrite('red_channel.jpg', image_contour_red)
# cv2.destroyAllWindows()
