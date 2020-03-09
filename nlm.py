import cv2, sys

##############################################################

# These parameter values are indicative. You should choose your own
# according to properties of the method you want to demonstrate
h = 5
templateWindowSize = 7
searchWindowSize = 21


if len(sys.argv) == 5:
    h = int(sys.argv[2])
    templateWindowSize = int(sys.argv[3])
    searchWindowSize = int(sys.argv[4])


##############################################################


img = cv2.imread(sys.argv[1])

dst = cv2.fastNlMeansDenoisingColored(img, None, h, h, templateWindowSize, 21)

cv2.imwrite('denoised.png', dst)


