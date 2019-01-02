import numpy as np
import argparse
import cv2
import imutils 

#from skimage.exposure import rescale_intensity
# construct average blurring kernels used to smooth an image
smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))

# construct a sharpening filter
sharpen = np.array((
	[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]), dtype="int")

# construct the Laplacian kernel used to detect edge-like
# regions of an image
laplacian = np.array((
	[0, 1, 0],
	[1, -4, 1],
	[0, 1, 0]), dtype="int")

# construct the Sobel x-axis kernel
sobelX = np.array((
	[-1, 0, 1],
	[-2, 0, 2],
	[-1, 0, 1]), dtype="int")

# construct the Sobel y-axis kernel
sobelY = np.array((
	[-1, -2, -1],
	[0, 0, 0],
	[1, 2, 1]), dtype="int")

def convolve(image, kernel):
	# grab the spatial dimensions of the image, along with
	# the spatial dimensions of the kernel
	(iH, iW) = image.shape[:2]
	(kH, kW) = kernel.shape[:2]

	# allocate memory for the output image, taking care to
	# "pad" the borders of the input image so the spatial
	# size (i.e., width and height) are not reduced
	pad = (kW - 1) // 2
	image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
		cv2.BORDER_REPLICATE)
	output = np.zeros((iH, iW), dtype="float32")

	# loop over the input image, "sliding" the kernel across
	# each (x, y)-coordinate from left-to-right and top to
	# bottom
	for y in np.arange(pad, iH + pad):
		for x in np.arange(pad, iW + pad):
			# extract the ROI of the image by extracting the
			# *center* region of the current (x, y)-coordinates
			# dimensions
			roi = image[y - pad:y + pad + 1, x - pad:x + pad + 1]

			# perform the actual convolution by taking the
			# element-wise multiplicate between the ROI and
			# the kernel, then summing the matrix
			k = (roi * kernel).sum()

			# store the convolved value in the output (x,y)-
			# coordinate of the output image
			output[y - pad, x - pad] = k

	# rescale the output image to be in the range [0, 255]
	output = rescale_intensity(output, in_range=(0, 255))
	output = (output * 255).astype("uint8")

	# return the output image
	return output

def img_hsv_range(img):
    img_hsv=cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0+mask1

    output_img = img.copy()
    output_img[np.where(mask==0)] = 0
    return output_img

def img_color_range(image):
    lower_red, upper_red = ([0, 0, 10], [100, 100, 200])
    lower = np.array(lower_red, dtype = "uint8")
    upper = np.array(upper_red, dtype = "uint8")
 
    mask = cv2.inRange(image, lower, upper)
    output_img = cv2.bitwise_and(image, image, mask = mask)
    
    
    return output_img

def seg_img(img, fn):
    #kernel = sharpen
    #convoleOutput = convolve(img, kernel)
    #filterOutput = cv2.filter2D(img, -1, kernel)
    #cv2.imwrite("filter-" + fn, filterOutput)
    #img = filterOutput
    output_img = img_hsv_range(img)
    
    #cv2.imwrite("images-out-"+fn, output_img)
    
    #gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    # find contours in the thresholded image, then initialize the
    # list of group locations
    groupCnts = cv2.findContours(gray.copy(),
        #cv2.RETR_EXTERNAL, 
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)

    groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
    groupLocs = []

    clone = np.dstack([gray.copy()] * 3)
    def union(a,b):
        x = min(a[0], b[0])
        y = min(a[1], b[1])
        w = max(a[0]+a[2], b[0]+b[2]) - x
        h = max(a[1]+a[3], b[1]+b[3]) - y
        return (x, y, w, h)

    def intersection(a,b):
        x = max(a[0], b[0])
        y = max(a[1], b[1])
        w = min(a[0]+a[2], b[0]+b[2]) - x
        h = min(a[1]+a[3], b[1]+b[3]) - y
        if w < 0 or h < 0: 
            return ()
        return (x, y, w, h)
    # loop over the group contours
    for (i, c) in enumerate(groupCnts):
        # compute the bounding box of the contour    
        a = cv2.boundingRect(c)
        # only accept the contour region as a grouping of characters if
        # the ROI is sufficiently large
        if a[2] < 50 or a[3] < 50:
            continue
        for (j, d) in enumerate(groupCnts):
        # compute the bounding box of the contour
            if j == i:
                continue
            b = cv2.boundingRect(d)
            inter = intersection(a, b)
            if len(inter) > 0:
                a = union(a, b)
        groupLocs.append(a)
    groupRec = []
    for i in range(len(groupLocs)):
        found = False
        for j in range(len(groupRec)):
            if groupLocs[i][0] == groupRec[j][0] and groupLocs[i][1] == groupRec[j][1] and groupLocs[i][2]== groupRec[j][2] and groupLocs[i][3] == groupRec[j][3]:
                found = True
                break
        if not found:
            groupRec.append(groupLocs[i])

    return clone, groupRec

def test_it():
    parser = argparse.ArgumentParser(description='color segmentation')
    parser.add_argument('--input', help='Path to input image.', default='image.jpg')
    args = parser.parse_args()
    img = cv2.imread(args.input)
    if img is None:
        print('Could not open or find the image:', args.input)
        exit(0)
    
    srcClone = img.copy()
    fn = args.input
    #img2 = render(img.copy())
    #cv2.imwrite("render-"+fn, img2)
    _, locs = seg_img(img, fn)
    c = 0
    for i in locs:
        localClone = img.copy()
        cv2.rectangle(localClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,0,0), 10)
        #cv2.imwrite("seg-" +str(c) + "-" + fn, localClone)
        cv2.rectangle(srcClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,0,0), 10)
        c = c + 1
    print locs
    cv2.imwrite("seg-all-" + fn, srcClone)

if __name__ == '__main__':
    test_it()