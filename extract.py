import numpy as np
import argparse
import cv2
import imutils 

def seg_img(img, fn):
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
    cv2.imwrite("images-out-"+fn, output_img)

    output_hsv = img_hsv.copy()
    output_hsv[np.where(mask==0)] = 0
    cv2.imwrite("images-hsv-"+fn, output_hsv)
    
    gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)

    #kernel = np.ones((5,5), np.uint8)
    #img_erode = cv2.erode(gray.copy(), kernel, iterations=1)
    #dilate = cv2.dilate(img_erode, kernel, iterations=1)
    #blur = cv2.bilateralFilter(dilate,9,75,75)
    #cv2.imwrite("dilate-" + fn, dilate)

    blur = cv2.GaussianBlur(gray,(3,3), 0)
    thresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_MEAN_C,\
                cv2.THRESH_BINARY_INV, 7,10)
    # find contours in the thresholded image, then initialize the
    # list of group locations
    clone = np.dstack([gray.copy()] * 3)

    groupCnts = cv2.findContours(thresh.copy(),  
        #cv2.RETR_EXTERNAL, 
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_NONE)
    #print(h)
    groupCnts = groupCnts[0] if imutils.is_cv2() else groupCnts[1]
    groupLocs = []

    clone = np.dstack([gray.copy()] * 3)
    # loop over the group contours
    for (i, c) in enumerate(groupCnts):
        # compute the bounding box of the contour
        (x, y, w, h) = cv2.boundingRect(c)
        # only accept the contour region as a grouping of characters if
        # the ROI is sufficiently large
        found = False
        for (j, d) in enumerate(groupCnts):
        # compute the bounding box of the contour
            (x1, y1, w1, h1) = cv2.boundingRect(d)
            if i== j:
                continue
            if x1 <= x and x1 + w1 >= x and y1 <= y and y1 + h1 >=  y :
                if x + w >= x1 and x + w <= x1 + w1 and y + h >= y1 and y1 + h1 >= y + h: 
                    found = True
                    break
        if found == False:
            cv2.rectangle(clone, (x,y), (x+w, y+h), (255,0,0), 1)
            #groupLocs.append((x, y, w, h))

    cv2.imwrite("seg-" + fn, clone)

parser = argparse.ArgumentParser(description='color segmentation')
parser.add_argument('--input', help='Path to input image.', default='image.jpg')
args = parser.parse_args()
img = cv2.imread(args.input)
if img is None:
    print('Could not open or find the image:', args.input)
    exit(0)
fn = args.input
seg_img(img, fn)