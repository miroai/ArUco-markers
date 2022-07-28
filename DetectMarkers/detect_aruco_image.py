#!/usr/bin/env python

# import the necessary packages
import argparse, cv2, os
from PIL import Image
import numpy as np

# define names of each possible ArUco tag OpenCV supports
ARUCO_DICT = {
	"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
	"DICT_4X4_100": cv2.aruco.DICT_4X4_100,
	"DICT_4X4_250": cv2.aruco.DICT_4X4_250,
	"DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
	"DICT_5X5_50": cv2.aruco.DICT_5X5_50,
	"DICT_5X5_100": cv2.aruco.DICT_5X5_100,
	"DICT_5X5_250": cv2.aruco.DICT_5X5_250,
	"DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
	"DICT_6X6_50": cv2.aruco.DICT_6X6_50,
	"DICT_6X6_100": cv2.aruco.DICT_6X6_100,
	"DICT_6X6_250": cv2.aruco.DICT_6X6_250,
	"DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
	"DICT_7X7_50": cv2.aruco.DICT_7X7_50,
	"DICT_7X7_100": cv2.aruco.DICT_7X7_100,
	"DICT_7X7_250": cv2.aruco.DICT_7X7_250,
	"DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
	"DICT_ARUCO_ORIGINAL": cv2.aruco.DICT_ARUCO_ORIGINAL,
#	"DICT_APRILTAG_16h5": cv2.aruco.DICT_APRILTAG_16h5,
#	"DICT_APRILTAG_25h9": cv2.aruco.DICT_APRILTAG_25h9,
#	"DICT_APRILTAG_36h10": cv2.aruco.DICT_APRILTAG_36h10,
#	"DICT_APRILTAG_36h11": cv2.aruco.DICT_APRILTAG_36h11
}

def AnnotateMarkers(im_path, tag_type):
	# load the input image from disk and resize it
	print(f"[INFO] loading image from {im_path}...")
	image = cv2.imread(im_path)
	# image = imutils.resize(image, width=600)

	# verify that the supplied ArUCo tag exists and is supported by
	# OpenCV
	assert tag_type in ARUCO_DICT.keys(), \
		f"[INFO] ArUCo tag of '{tag_type}' is not a supported type: {list(ARUCO_DICT.keys())}"

	# load the ArUCo dictionary, grab the ArUCo parameters, and detect
	# the markers
	print(f"[INFO] detecting '{tag_type}' tags...")
	arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[tag_type])
	arucoParams = cv2.aruco.DetectorParameters_create()
	(corners, ids, rejected) = cv2.aruco.detectMarkers(image, arucoDict,
		parameters=arucoParams)

	# verify *at least* one ArUco marker was detected
	if len(corners) < 1:
		print(f'no tags found for {im_path}')
		return None
	else:
		# flatten the ArUco IDs list
		ids = ids.flatten()

		# loop over the detected ArUCo corners
		for (markerCorner, markerID) in zip(corners, ids):
			# extract the marker corners (which are always returned in
			# top-left, top-right, bottom-right, and bottom-left order)
			corners = markerCorner.reshape((4, 2))
			(topLeft, topRight, bottomRight, bottomLeft) = corners

			# convert each of the (x, y)-coordinate pairs to integers
			topRight = (int(topRight[0]), int(topRight[1]))
			bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
			bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
			topLeft = (int(topLeft[0]), int(topLeft[1]))

			# draw the bounding box of the ArUCo detection
			cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
			cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
			cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
			cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

			# compute and draw the center (x, y)-coordinates of the ArUco
			# marker
			cX = int((topLeft[0] + bottomRight[0]) / 2.0)
			cY = int((topLeft[1] + bottomRight[1]) / 2.0)
			cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)

			# draw the ArUco marker ID on the image
			cv2.putText(image, str(markerID),
				(topLeft[0], topLeft[1] - 15), cv2.FONT_HERSHEY_SIMPLEX,
				0.5, (0, 255, 0), 2)
			print("[INFO] ArUco marker ID: {}".format(markerID))

		# # show the output image
		# cv2.imshow("Image", image)
		# cv2.waitKey(0)
		return image

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description= f'Image ArUco markers detection')
	parser.add_argument("-i", "--image", required=True, type = str,
						help="path to input image containing ArUCo tag")
	parser.add_argument("-t", "--type", type=str,
						default="DICT_ARUCO_ORIGINAL",
						help="type of ArUCo tag to detect [default: DICT_ARUCO_ORIGINAL]")
	parser.add_argument('--out_path', required = True, type = str,
							help = 'output Image path')
	args = parser.parse_args()
	assert os.path.isfile(args.image), f'{args.image} is not a valid file'
	assert os.path.isdir(os.path.dirname(args.out_path)), f'output directory: {os.path.dirname(args.out_path)} not found!'

	im_bgr = AnnotateMarkers(args.image, tag_type = args.type)
	assert isinstance(im_bgr, np.ndarray), f'Function AnnotateMarkers() found no markers :('

	im = Image.fromarray(im_bgr[:,:,::-1])
	im.save(args.out_path)
	print(f'ArUco Markers annotation saved to {args.out_path}')
