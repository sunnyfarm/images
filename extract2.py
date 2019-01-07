import numpy as np
import argparse
import cv2
import imutils 

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

def merge(groupCnts, max_w, max_h):
    groupLocs = []
    for (i, c) in enumerate(groupCnts):
        # compute the bounding box of the contour    
        a = cv2.boundingRect(c)
        # only accept the contour region as a grouping of characters if
        # the ROI is sufficiently large
        if a[2] < 50 or a[3] < 50 or a[2] > max_w or a[3] > max_h:
            continue
        for (j, d) in enumerate(groupCnts):
        # compute the bounding box of the contour
            if j == i:
                continue
            b = cv2.boundingRect(d)
            if b[2] < 50 or b[3] < 50 or b[2] > max_w or b[3] > max_h:
                continue
            
            inter = intersection(a, b)
            if len(inter) > 0:
                a = union(a, b)
#            if x1 <= x and x1 + w1 >= x and y1 <= y and y1 + h1 >=  y :
#                if x + w >= x1 and x + w <= x1 + w1 and y + h >= y1 and y1 + h1 >= y + h: 
#                    found = True
#                    break
        #if found == False:
        #    groupLocs.append(a)
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

    return groupRec

def hsv_img(img, fn):
    my_img = img #cv2.pyrDown(img)
    img_hsv=cv2.cvtColor(my_img, cv2.COLOR_BGR2HSV)

    # lower mask (0-10)
    lower_red = np.array([0,50,50])
    upper_red = np.array([10,255,255])
    mask0 = cv2.inRange(img_hsv, lower_red, upper_red)

    # upper mask (170-180)
    lower_red = np.array([170,50,50])
    upper_red = np.array([180,255,255])
    mask1 = cv2.inRange(img_hsv, lower_red, upper_red)

    mask = mask0+mask1
    output_img = my_img
    output_img[np.where(mask==0)] = 0
    #cv2.imwrite("images-out-"+fn, output_img)

    #output_hsv = img_hsv.copy()
    #output_hsv[np.where(mask==0)] = 0
    #cv2.imwrite("images-hsv-"+fn, output_hsv)
    
    #gray = cv2.cvtColor(output_hsv, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(output_img, cv2.COLOR_BGR2GRAY)
    # find contours in the thresholded image, then initialize the
    # list of group locations
    #groupCnts =  get_contours(img.copy())

    clone = np.dstack([gray.copy()] * 3)
    # loop over the group contours
    #groupRec = merge(groupCnts)
    return clone #, groupRec

def seg_img(img, fn):
    clone = hsv_img(img.copy(), fn)
    height, width, channels = clone.shape
    contours = get_contours(img.copy())
    groupRec = merge(contours, width/2, height/2)
    return clone, groupRec
    
def box(img):
    contours = get_contours(img.copy())

    for (idx, contour) in enumerate(contours):
        x, y, w, h = cv2.boundingRect(contour)
       
        if w > 20 and h > 20:
            cv2.rectangle(img, (x, y), ((x+w-1), (y+h-1)), (0, 255, 0), 2)

    return img

def box2(img):
    contours = get_contours(img.copy())
    height, width, channels = img.shape
    contours = get_contours(img.copy())
    groupRec = merge(contours, width/2, height/2)

    return groupRec

def get_contours(large):    
    rgb = cv2.pyrDown(large.copy())
    try:
        small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    except:
        small = rgb

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = contours[0] if imutils.is_cv2() else contours[1]
    
    for (idx, contour) in enumerate(contours):
        contours[idx] = contour * 2

    return contours


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
    image, locs = seg_img(img.copy(), fn)
    c = 0
    for i in locs:
        im = image[i[1]:i[1] + i[3], i[0]:i[0] + i[2]]   
        cv2.imwrite("seg-" +str(c) + "-" + fn, im)
        cv2.rectangle(srcClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,0,0), 10)
        c = c + 1
    print(locs)
    cv2.imwrite("seg-all-" + fn, srcClone)
    rgb = box(img.copy())
    cv2.imwrite("box-" + fn, rgb)

    srcClone = img.copy()    
    locs = box2(img.copy())
    c = 0
    for i in locs:
        cv2.rectangle(srcClone, (i[0], i[1]), (i[0] + i[2], i[1] + i[3]), (0,0,0), 10)
    cv2.imwrite("box2-" + fn, srcClone)

        
if __name__ == '__main__':
    test_it()
    
