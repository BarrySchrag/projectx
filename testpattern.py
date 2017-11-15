# A class which creates and updates a test pattern
# Update function acts on the test patterns in the chosen manner

from math import sqrt, acos, pi
import cv2
import numpy as np
import imutils

# class Helper():
#
# 	x = None
# 	y = None
#
# 	def __init__(self, x, y):
# 		self.x, self.y = x, y
#
# 	def __str__(self):
# 		return "%6.1f, %6.1f" % (self.x, self.y)
#
# 	def __eq__(self, obj):
# 		return obj.x == self.x and obj.y == self.y
#
# 	def distance_to_point(self, p):
# 		return sqrt((self.x-p.x)**2+(self.y-p.y)**2)

def createTransforms(offset):
	R = np.float32 ( [[1, 0, offset], [0, 1, 0]] )
	L = np.float32 ( [[1, 0, -offset], [0, 1, 0]] )
	U = np.float32 ( [[1, 0, 0], [0, 1, -offset]] )
	D = np.float32 ( [[1, 0, 0], [0, 1, -offset]] )
	return R, L, U, D

class TestPattern():

	# Screen coordinates
	l_top  = None
	r_top  = None
	l_bot  = None
	r_bot  = None
	center = None
	width  = None
	height = None

	def __init__(self, bg_image, bg_width, bg_height, fg_image, fg_width, fg_height, ):
		assert width>0
		assert height>0
		self.l_top  = Point(x, y)
		self.r_top  = Point(x+width, y)
		self.r_bot  = Point(x+width, y+height)
		self.l_bot  = Point(x, y+height)
		self.center = Point(x+width/float(2), y+height/float(2))
		self.width  = width
		self.height = height


	# Gives back a copy of this rectangle
	def copy(self):
		return Rect(self.l_top.x, self.l_top.y, self.width, self.height)

	overlay_width = 60

	bg = cv2.imread ( 'background_image.jpg', 0 )
	bgrows, bgcols = bg.shape

	overlay = cv2.imread ( overlay_image.png
	')
	overlayrows, overlaycols = overlay.shape

	# resize the overlay
	overlay = imutils.resize ( overlay, overlay_width )
	offset_bg = 1
	offset_fg = 1
	bgMotion = createTransforms ( offset_bg )
	fgMotion = createTransforms ( offset_fg )

	while True:
	# extract the right most column
	bgrightcol = bg[0:bgrows, bgcols - 1:bgcols]  # [y:y+h, x:x+w]

	# warp the image right
	bgdst = cv2.warpAffine ( img, bgMotion[0], (bgcols, bgrows) )

	# copy the column to the left side
	bgdst[0:rbgows, 0, 0] = bgrightcol

# ---------------------- Static functions ----------------------

#def distance_between_points(point1, point2):
#	return point1.distance_to_point(point2)
