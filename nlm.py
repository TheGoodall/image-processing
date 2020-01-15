import cv2

##############################################################

# These parameter values are indicative. You should choose your own
# according to properties of the method you want to demonstrate

h = 5
templateWindowSize = 7
searchWindowSize = 21

##############################################################

import sys

img = cv2.imread('Images/'+sys.argv[1])

dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, 21)

cv2.imwrite('denoised.png', dst)


