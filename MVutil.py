import cv2
import numpy as np
from datetime import datetime

# Main for testing
def main():
    print("hello world")
    
    img = cv2.imread("fruits.jpg")
    
    cv2.imshow("input", img)
    
    new_img = gammaCorrection(img, 0.2)
    
    cv2.imshow("output", new_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

## 
# Rotates the provided image by the provided degrees
# @param img The image to be rotated
# @param deg The amount in degrees 
# \return The rotated image.
def rotateImage(img, deg):
    (h, w) = img.shape[:2]
    center = (w/2, h/2)
    
    M = cv2.getRotationMatrix2D(center, deg, 1.0)
    new_img = cv2.warpAffine(img, M, (w, h))
    
    return new_img
   
## 
# Places a timestamp on the provided image at (x, y),
# with additional options to modify the font, size, color
# and type of line the text is drawn in
# @param img The image to be drawn on
# @param x The x location of where the text will begin writing
# @param y The y location of where the text will begin writing
# @param font The font to be used for text (DEFAULT = HERSHEY SIMPLEX)
# @param font_scale The scale of the text drawn (DEFAULT = 1.0)
# @param font_color The color in which the text is drawn (DEFAULT = (255, 255, 255))
# @param line_type The type of line to be used while drawing text (DEFAULT = 2)
# \return The image with the timestamp drawn on it.
def timestamp(img, x, y, font = cv2.FONT_HERSHEY_SIMPLEX,
              font_scale = 1.0, font_color = (255, 255, 255),
              line_type = 2):
    
    now = datetime.now()
    dt_str = now.strftime("%m/%d/%Y %H:%M:%S")
    
    new_img = cv2.putText(img, dt_str, (x, y),
        font, font_scale, font_color, line_type)
    
    return new_img

##
# Writes the text on the provided image at (x, y),
# with additional options to modify the font, size, color
# and type of line the text is drawn in
# @param img The image to be drawn on
# @param x The x location of where the text will begin writing
# @param y The y location of where the text will begin writing
# @param text The text to be written
# @param font The font to be used for text (DEFAULT = HERSHEY SIMPLEX)
# @param font_scale The scale of the text drawn (DEFAULT = 1.0)
# @param font_color The color in which the text is drawn (DEFAULT = (255, 255, 255))
# @param line_type The type of line to be used while drawing text (DEFAULT = 2)
# \return The image with the text drawn on it.
def writeText(img, x, y, text, font = cv2.FONT_HERSHEY_SIMPLEX,
			  font_scale = 1.0, font_color = (255, 255, 255),
              line_type = 2):
			  
	new_img = cv2.putText(img, text, (x, y),
        font, font_scale, font_color, line_type)
		
	return new_img
	
	
## 
# Crops the provided image, starting at (x, y) and 
# going w pixels in width, and h pixels in height
# @param img The image to be cropped
# @param x The starting x location to begin cropping
# @param y The starting y location to begin cropping
# @param w The horizontal distance from x in which to crop
# @param h The vertical distance from y in which to crop
# \return The cropped image.
def cropImage(img, x, y, w, h):
    new_img = img[y:y+h, x:x+h]
    return new_img

## 
# Changes the contrast of the provided image using
# the passed in value of alpha
# @param img The image to be modified
# @param alpha The amount in which the contrast will be changed
# \return The image with modifed contrast.
def changeContrast(img, alpha):
    new_img = np.zeros_like(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                new_pixel = img[i,j,k] * alpha
                if new_pixel < 0: new_pixel = 0
                if new_pixel > 255: new_pixel = 255
    
                new_img[i,j,k] = int(new_pixel)
                
    return new_img

## 
# Changes the brightness of the provided image using
# the passed in value of beta
# @param img The image to be modified
# @param beta The amount in which the brightness will be changed
# \return The image with modified brightness.
def changeBrightness(img, beta):
    new_img = np.zeros_like(img)
    for i in range(len(img)):
        for j in range(len(img[0])):
            for k in range(len(img[0][0])):
                new_pixel = img[i,j,k] + beta
                if new_pixel < 0: new_pixel = 0
                if new_pixel > 255: new_pixel = 255
    
                new_img[i,j,k] = int(new_pixel)
    
    return new_img

##	
# Performs gamma correction on the provided image
# using the passed in value of gamma
# @param img The image to be modifed
# @param gamma The gamma correction parameter
# \return The gamma corrected image.
def gammaCorrection(img, gamma):
    gamma_inv = 1.0 / gamma
    table = np.array([((i / 255.0) ** gamma_inv) * 255
        for i in np.arange(0, 256)]).astype("uint8")
    new_img = cv2.LUT(img, table)
    
    return new_img
    
##
# Takes in 2 images, alongside their keypoints and descriptors
# and draws the matches between the two, matches the points,
# and draws the best matches, the number of which is determined
# by num_matches
# @param img1 The first image to match
# @param img2 The second image to match
# @param kp1 The first image's keypoints
# @param des1 The first image's descriptors
# @param kp2 The second image's keypoints
# @param des2 The second image's descriptors
# @param num_matches The number of (best) matches to draw
# \return An image consisting of img1 and img2, with matches drawn between them.
def matchDes(img1, img2, kp1, des1, kp2, des2, num_matches):
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	
	# Match descriptors.
	matches = bf.match(des1,des2)
	
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	
	# Draw first so many matches.
	output_frame = cv2.drawMatches(img1,kp1,img2,kp2,matches[:num_matches], None, flags=2)
	
	return output_frame
	
##
# Takes in 2 images, alongside their keypoints and descriptors
# and draws the matches between the two, matches the points,
# and using num_matches to determine how many matches to use,
# attempts to draw a box around the contents of the first image
# in the second image
# @param img1 The first image to match
# @param img2 The second image to match
# @param kp1 The first image's keypoints
# @param des1 The first image's descriptors
# @param kp2 The second image's keypoints
# @param des2 The second image's descriptors
# @param num_matches The number of (best) matches to draw
# \return An image consisting of img1 and img2, with matches drawn between them, as well as a bounding box on the second image, representing the first image.
def matchDesBox(img1, img2, kp1, des1, kp2, des2, num_matches):
	# create BFMatcher object
	bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
	
	# Match descriptors.
	matches = bf.match(des1,des2)
	
	# Sort them in the order of their distance.
	matches = sorted(matches, key = lambda x:x.distance)
	
	good_matches = matches[:10]
	
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good_matches ]).reshape(-1,1,2)
	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good_matches ]).reshape(-1,1,2)
	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	matchesMask = mask.ravel().tolist()
	h,w = img1.shape[:2]
	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	
	dst = cv2.perspectiveTransform(pts,M)
	dst += (w, 0)  # adding offset
	
	draw_params = dict(matchColor = (0,255,0), # draw matches in green color
               singlePointColor = None,
               matchesMask = matchesMask, # draw only inliers
               flags = 2)

	img3 = cv2.drawMatches(img1,kp1,img2,kp2,good_matches, None,**draw_params)

	# Draw bounding box in Red
	img3 = cv2.polylines(img3, [np.int32(dst)], True, (0,0,255),3, cv2.LINE_AA)

	return img3

##
# Takes an image and performs blob detection to produce an image
# with keypoints drawn on, as well as the logical keypoints and
# descriptors
# @param image The image to be used in blob detection
# \return The image with keypoints drawn on, the keypoints, and the descriptors.
def blobDetection(image):
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	
	detector = cv2.SimpleBlobDetector_create()
	
	# Use brisk to compute descriptors
	brisk = cv2.BRISK_create()
	
	kp = detector.detect(gray)
	kp, des = brisk.compute(image, kp)
	
	im_with_keypoints = cv2.drawKeypoints(image, kp, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

	#cv2.imshow("Keypoints", im_with_keypoints)
	return im_with_keypoints, kp, des

##
# Takes an image and performs FAST feature detection to produce
# an image with keypoints drawn on, as well as the logical 
# keypoints and descriptors	
# @param image The image to be used in FAST detection
# \return The image with keypoints drawn on, the keypoints, and the descriptors.
def FAST(image):
	fast = cv2.FastFeatureDetector_create()
	# Use brisk to compute descriptors
	brisk = cv2.BRISK_create()
	
	# find and draw the keypoints
	kp = fast.detect(image,None)
	# Compute descriptors
	kp, des = brisk.compute(image, kp)
	
	im_with_keypoints = image
	im = cv2.drawKeypoints(image, kp, im_with_keypoints, color=(255,0,0))

	return im_with_keypoints, kp, des
	
##
# Takes an image and performs ORB feature detection to produce
# an image with keypoints drawn on, as well as the logical
# keypoints and descriptors
# @param image The image to be used in ORB detection
# \return The image with keypoints drawn on, the keypoints, and the descriptors.
def ORB(image):
	# Initiate STAR detector
	orb = cv2.ORB_create()
	
	# find the keypoints with ORB
	kp = orb.detect(image,None)
	
	# compute the descriptors with ORB
	kp, des = orb.compute(image, kp)
	
	im_with_keypoints = image
	# draw only keypoints location,not size and orientation
	im_with_keypoints = cv2.drawKeypoints(image, kp, im_with_keypoints, color=(0,255,0), flags=0)
	
	return im_with_keypoints, kp, des
	
##
# Takes an image and performs BRISK feature detection to produce 
# an image with keypoints drawn on, as well as the logical 
# keypoints and descriptors
# @param image The image to be used in BRISK detection
# \return The image with keypoints drawn on, the keypoints, and the descriptors.
def BRISK(image):
	brisk = cv2.BRISK_create()
	
	kp = brisk.detect(image)
	kp, des = brisk.compute(image, kp)
	
	im_with_keypoints = image
	
	im_with_keypoints = cv2.drawKeypoints(image, kp, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	return im_with_keypoints, kp, des

##
# An attempt at using SIFT feature detection, will not work
# on the standard installation of openCV, due to SIFT being
# an non-free library. Requires additional packages to work
# @param image The image to be used in SIFT detection
# \return The image with keypoints drawn on, the keypoints, and the descriptors.
def SIFT(image):
	sift = cv2.SIFT()
	
	kp = sift.detect(image, None)
	kp, des = sift.compute(image, kp)
	
	im_with_keypoints = image
	im_with_keypoints = cv2.drawKeypoints(image, kp, im_with_keypoints, color=(0,255,0), flags=0)
	
	return im_with_keypoints, kp, des
	
##
# Begins camera calibration, using a video capture. Requires a checkerboard calibration board with no less than 8 by 6 set of squares.
# @param num_good_points The number of valid calibration points to be taken before the calibration occurs
# \return The mtx, dist, tvecs, and rvecs, required for undistortion.
def beginCalibration(num_good_points):
	cap = cv2.VideoCapture(0)
	
	
	# termination criteria
	criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
	
	
	# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
	objp = np.zeros((6*8,3), np.float32)
	objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
	
	# Arrays to store object points and image points from all the images.
	objpoints = [] # 3d point in real world space
	imgpoints = [] # 2d points in image plane.
	
	while(True):
		# Capture frame-by-frame
		ret, frame = cap.read()
		
		# Our operations on the frame come here
		
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		ret, corners = cv2.findChessboardCorners(gray, (8,6),None)
		# Display the resulting frame
		
		img = gray
		
		if ret == True:
			objpoints.append(objp)
			print("found corners; count: ", len(objpoints))
			
			corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
			imgpoints.append(corners2)
			
			# Draw and display the corners
			img = cv2.drawChessboardCorners(frame, (8,6), corners2,ret)
	
		cv2.waitKey(500)
		cv2.imshow('frame',img) 
    
		if len(objpoints) > num_good_points:
			ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
			break
    
	ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None, None)
	
	cap.release()
	cv2.destroyAllWindows()
	
	return mtx, dist, rvecs, tvecs

##
# Undistorts the passed in image and returns it
# @param img The image to be undistorted
# @param mtx The mtx value from camera calibration
# @param dist The dist value from camera calibration
# \return The undistorted image.
def undistort(img, mtx, dist):
	h, w = img.shape[:2]
	newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
	
	new_img = cv2.undistort(img, mtx, dist, None, newcameramtx)
	
	return new_img
	
##
# Saves calibration results in files for future calibration
# @param mtx The mtx values to be saved
# @param dist The dist values to be saved
# @param rvecs The rvecs values to be saved
# @param tvecs The tvecs values to be saved
# \return Nothing.
def saveCamMat(mtx, dist, rvecs, tvecs):
	np.save("mtx", mtx)
	np.save("dist", dist)
	np.save("rvecs", rvecs)
	np.save("tvecs", tvecs)

##
# Simply reads the saved matricies and loads them back
# into their variables (and returns them)
# \return Mtx, dist, rvecs, tvecs to be used in undistortion.
def loadCamMat():
    mtx = np.load("mtx.npy")
    dist = np.load("dist.npy")
    rvecs = np.load("rvecs.npy")
    tvecs = np.load("tvecs.npy")
    
    return mtx, dist, rvecs, tvecs
	
##
# Applies the calibration relative to the passed in image/frame
# and returns calibrated matrix and roi
# @param img The image to undistort
# @param mtx The mtx values required to undistort
# @param dist The dist values required to undistort
# \return The calibrated camera matrix as well as the roi
def applyCalibration(img, mtx, dist):
    h, w = img.shape[::2]
    newcameramtx, roi = cv2.getOptimaNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
    
    return newcameramtx, roi
	
# Main for testing
if __name__ == "__main__":
    main()