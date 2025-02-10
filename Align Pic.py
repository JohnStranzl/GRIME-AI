# THIS
# REFERENCE TALKS ABOUT THE ISSUES CREATED BY NEWER VERSIONS OF OPENCV WHEN SORTING
# https://stackoverflow.com/questions/73157702/attributeerror-tuple-object-has-no-attribute-sort

# JOHN E. STRANZL JR.
# GRIME-AI
# Copyright 2024

import os

import cv2
import numpy as np 

os.chdir("C:\\Users\\johns\\pycharmprojects\\neonAI")

# Open the image files. 
img1_color = cv2.imread("NM_Pecos_River_near_Acme___2023-08-17T00-00-46Z.jpg")  # Image to be aligned. 
#img2_color = cv2.imread("NM_Pecos_River_near_Acme___2023-12-19T23-30-07Z.jpg")  # Reference image. 
img2_color = cv2.imread("NM_Pecos_River_near_Acme___2023-12-19T23-30-07Z.jpg.tmp.jpg")  # Reference image. 

# Convert to grayscale. 
img1 = cv2.cvtColor(img1_color, cv2.COLOR_BGR2GRAY) 
img2 = cv2.cvtColor(img2_color, cv2.COLOR_BGR2GRAY) 
height, width = img2.shape 

cv2.imshow("", img1)
cv2.imshow("", img2)

# Create ORB detector with 5000 features. 
orb_detector = cv2.ORB_create(5000) 
  
# Find keypoints and descriptors. 
# The first arg is the image, second arg is the mask 
#  (which is not required in this case). 
kp1, d1 = orb_detector.detectAndCompute(img1, None) 
kp2, d2 = orb_detector.detectAndCompute(img2, None) 
  
# Match features between the two images. 
# We create a Brute Force matcher with  
# Hamming distance as measurement mode. 
matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True) 
  
# Match the two sets of descriptors. 
matches = matcher.match(d1, d2) 
  
# Sort matches on the basis of their Hamming distance. 
#matches.sort(key = lambda x: x.distance)
matches = sorted(matches, key = lambda x: x.distance)

# Take the top 90 % matches forward. 
matches = matches[:int(len(matches)*0.9)]
no_of_matches = len(matches) 
  
# Define empty matrices of shape no_of_matches * 2. 
p1 = np.zeros((no_of_matches, 2)) 
p2 = np.zeros((no_of_matches, 2)) 
  
for i in range(len(matches)): 
  p1[i, :] = kp1[matches[i].queryIdx].pt 
  p2[i, :] = kp2[matches[i].trainIdx].pt 
  
# Find the homography matrix. 
homography, mask = cv2.findHomography(p1, p2, cv2.RANSAC) 

# Compute rotation angle based on matched keypoints
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
rotation_angle = np.arctan2(M[1, 0], M[0, 0]) * 180 / np.pi

# Use this matrix to transform the colored image wrt the reference image. 
transformed_img = cv2.warpPerspective(img1_color, homography, (width, height)) 
  
# Save the output. 
cv2.imwrite('output.jpg', transformed_img)

cv2.waitKey(0)

# Print rotation angle
print("Rotation angle:", rotation_angle)

# import cv2
# import numpy as np

# def detect_rotation(reference_img_path, img_path):
#     # Load reference image
#     ref_img = cv2.imread(reference_img_path)
#     ref_img = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
#     ref_img = cv2.resize(ref_img, (500, 500))
#     cv2.imshow('Reference Image', ref_img)

#     # Load image to be compared
#     #img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
#     img = cv2.imread(img_path)
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.resize(img, (500, 500))
#     cv2.imshow("Image", img)

#     # Calculate difference between reference and compared image
#     diff = cv2.absdiff(ref_img, img)

#     # Apply thresholding to highlight differences
#     _, thresholded = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)

#     # Find contours of the thresholded image
#     contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     cv2.drawContours(img, contours, -1, (0, 255, 0), 2)
#     cv2.imshow("Contours", img)

#     # Calculate rotation angle
#     if len(contours) > 0:
#         rect = cv2.minAreaRect(contours[0])
#         angle = rect[2]
#         if angle < -45:
#             angle += 90
#     else:
#         angle = 0

#     cv2.waitKey(0)

#     return angle

# Path to reference image
# reference_img_path = "NM_Pecos_River_near_Acme___2023-08-17T00-00-46Z.jpg"

# Path to image to be compared
# img_path = "NM_Pecos_River_near_Acme___2023-12-19T23-30-07Z.jpg"

# Detect rotation
# rotation_angle = detect_rotation(reference_img_path, img_path)

# Print rotation angle
# print("Rotation angle:", rotation_angle)
