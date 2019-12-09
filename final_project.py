import cv2
import numpy as np
import copy
import MVutil as mvu
import time

#cap = cv2.VideoCapture(0)


def main():
	frame = cv2.imread("many1.png")
	bg = cv2.imread("many0.png")
	frame_sel = 1

	while(True):
		start = time.time()
		#ret, frame = cap.read()
		cv2.imshow("frame", frame)
		
		if cv2.waitKey(1) & 0xFF == ord('1'): frame_sel = 1
		if cv2.waitKey(1) & 0xFF == ord('2'): frame_sel = 2
		if cv2.waitKey(1) & 0xFF == ord('3'): frame_sel = 3
		if cv2.waitKey(1) & 0xFF == ord('4'): frame_sel = 4
		if cv2.waitKey(1) & 0xFF == ord('5'): frame_sel = 5
		if cv2.waitKey(1) & 0xFF == ord('6'): frame_sel = 6
		if cv2.waitKey(1) & 0xFF == ord('7'): frame_sel = 7

		
		if frame_sel == 1: frame = cv2.imread("many1.png")
		if frame_sel == 2: frame = cv2.imread("many2.png")
		if frame_sel == 3: frame = cv2.imread("many3.png")
		if frame_sel == 4: frame = cv2.imread("many4.png")
		if frame_sel == 5: frame = cv2.imread("many5.png")
		if frame_sel == 6: frame = cv2.imread("many6.png")
		if frame_sel == 7: frame = cv2.imread("many7.png")

		
		#print("fg - bg")
		
		# Convert background and foreground images to grayscale
		bg_gray = cv2.cvtColor(bg, cv2.COLOR_BGR2GRAY)
		frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		
		# Obtain the difference between fg and bg
		diff = cv2.absdiff(frame_gray, bg_gray)
		cv2.imshow("diff", diff)
		
		# Blur the difference
		blur = cv2.GaussianBlur(diff,(5,5),0)
		
		# Otsu Thresholding
		ret3, thresh_im = cv2.threshold(blur,100,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
		
		cv2.imshow("otsu", thresh_im)
		
		# Canny Edge detection
		edges = cv2.Canny(thresh_im,50,250)
		
		cv2.imshow("Canny edges", edges)
		
		# Find contours
		im2, contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

		hierarchy = hierarchy[0] # get the actual inner list of hierarchy descriptions
		cnt = contours[0]
		
		
		#frame_cpy1 = copy.deepcopy(frame)
		frame_cpy2 = copy.deepcopy(frame)
		
		bounding = frame_cpy2
		dice = []
		
		#print("contour size: ", len(contours))
		
		# Begin iterating through contours
		for c in contours:
		
			# Calculate area of contour, continue if it exceeds size
			c_area = cv2.contourArea(c)
			if c_area > 900:
				x,y,w,h = cv2.boundingRect(c)
				bounding = cv2.rectangle(bounding,(x,y),(x+w,y+h),(0,255,0),2)
				#print("x, y, w, h: ", x, y, w, h)
				
				# Crop section from main image
				cropped_die = mvu.cropImage(frame, x, y, w, h)
				
				dice.append(cropped_die)
				
				# Resize image
				
				size_die = cv2.resize(cropped_die, (0,0), fx = 2.0, fy = 2.0)
				dice.append(size_die)
				
				
				# Convert to grayscale
				gray_die = cv2.cvtColor(size_die, cv2.COLOR_BGR2GRAY)
				
				# Gaussian blur
				blur_die = cv2.GaussianBlur(gray_die,(5,5),0)
				dice.append(blur_die)
				
				# Otsu Thresholding
				ret, otsu_die = cv2.threshold(blur_die,140,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
				dice.append(otsu_die)

				die_h = otsu_die.shape[0]
				die_w = otsu_die.shape[1]
				fill_die = copy.deepcopy(otsu_die)

				#print("h, w: ", die_h, die_w)
				mask = np.zeros((die_h+2, die_w+2), np.uint8)
				
				# Bucket fill corners
				# Top-left
				cv2.floodFill(fill_die, mask, (0, 0), 255)
				# Top-right
				cv2.floodFill(fill_die, mask, (die_w-2, 0), 255)
				# Bottom-left
				cv2.floodFill(fill_die, mask, (0, die_h-2), 255)
				# Bottom-right
				cv2.floodFill(fill_die, mask, (die_w-2, die_h-2), 255)
				
				dice.append(fill_die)
				
				# Blob detection
				kp_die, kp = blob_detection(fill_die)
				
				
				dice.append(kp_die)
				#print(len(kp))

				# Use keypoints to count pips
				dice_num = len(kp)
				
				# Write number to dice location
				bounding = mvu.writeText(bounding, x, y, str(dice_num), font_color=(0, 0, 255))
		
		cv2.imshow("Bounding Boxes", bounding)
		
		if len(dice) > 0:
			cv2.imshow("Cropped Die", dice[0])
			cv2.imshow("Resized Die", dice[1])
			cv2.imshow("Blurred Die", dice[2])
			cv2.imshow("Otsu'ed Die", dice[3])
			cv2.imshow("Filled Die", dice[4])
			cv2.imshow("Blobs Die", dice[5])
		
		for i in range(5, len(dice), 6):
		
			cv2.imshow("Blob'ed Die"+str(int(i/6)), dice[i])			
		
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break
		
		end = time.time()
		
		seconds = end - start
		
		print("loop took: ", seconds)
	
	
	#cap.release()
	cv2.destroyAllWindows()
	
def blob_detection(image):
	#gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


	# Setup SimpleBlobDetector parameters.
	params = cv2.SimpleBlobDetector_Params()
	
	# Change thresholds
	params.minThreshold = 10;
	params.maxThreshold = 200;
	
	# This needs to be dependant on the image size I think
	# 
	# Filter by Area.
	params.filterByArea = True
	params.minArea = 50
	params.maxArea = 850
	
	# Filter by Circularity
	params.filterByCircularity = False
	params.minCircularity = 0.1
	
	# Filter by Convexity
	params.filterByConvexity = False
	params.minConvexity = 0.87
	
	# Filter by Inertia
	params.filterByInertia = False
	params.minInertiaRatio = 0.01
	
	detector = cv2.SimpleBlobDetector_create(params)
	
	keypoints = detector.detect(image)
	img_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	
	return img_with_keypoints, keypoints
	
if __name__ == "__main__":
	main()