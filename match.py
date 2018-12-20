#!/usr/bin/env python


# Python 2/3 compatibility
from __future__ import print_function
from pprint import pprint
import numpy as np
import cv2
from removebg import remove_bg
from extract import seg_img

FLANN_INDEX_KDTREE = 1  # bug: flann enums are missing
FLANN_INDEX_LSH    = 6


def init_feature(name):
    chunks = name.split('-')
    if chunks[0] == 'sift':
        detector = cv2.xfeatures2d.SIFT_create(edgeThreshold=5)
        norm = cv2.NORM_L2
    elif chunks[0] == 'surf':
        detector = cv2.xfeatures2d.SURF_create(800)
        norm = cv2.NORM_L2
    elif chunks[0] == 'orb':
        detector = cv2.ORB_create(nfeatures=10000, WTA_K=3, scoreType=cv2.ORB_FAST_SCORE)
        norm = cv2.NORM_HAMMING2
    elif chunks[0] == 'akaze':
        detector = cv2.AKAZE_create()
        norm = cv2.NORM_HAMMING
    elif chunks[0] == 'brisk':
        detector = cv2.BRISK_create()
        norm = cv2.NORM_HAMMING
    else:
        return None, None
    if 'flann' in chunks:
        if norm == cv2.NORM_L2:
            flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
        else:
            flann_params= dict(algorithm = FLANN_INDEX_LSH,
                               table_number = 6, # 12
                               key_size = 12,     # 20
                               multi_probe_level = 1) #2
        matcher = cv2.FlannBasedMatcher(flann_params, {})  # bug : need to pass empty dict (#1329)
    else:
        matcher = cv2.BFMatcher(norm)
    return detector, matcher


def filter_matches(kp1, kp2, matches, ratio = 0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append( kp1[m.queryIdx] )
            mkp2.append( kp2[m.trainIdx] )
        
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    
    return p1, p2

def match_and_draw(kp1, desc1, kp2, desc2, img1, img2, si, di):
    raw_matches = matcher.knnMatch(
        desc1, trainDescriptors=desc2, k=2)  # 2
    ratio = 0.85
    p1, p2 = filter_matches(kp1, kp2, raw_matches, ratio)
    good = []
    for m, n in raw_matches:
        if m.distance < ratio*n.distance:
            good.append([m])
    
    if len(p1) >= 2:
        h, status = cv2.findHomography(p1, p2, cv2.RANSAC) #, cv2.RANSAC, 5.0)
        inliner = []
        for i in range(status.size):
            if status[i] > 0:
                inliner.append(good[i])
        #h, status = cv2.findHomography(p1, p2, cv2.LMEDS)
        matched = np.sum(status) / len(status) * 100
        matches = np.sum(status)
        
        try:
            imgWarp = cv2.warpPerspective(img1, h, (img2.shape[1],img2.shape[0]))
        except:
            imgWarp = img2
        return matched, matches, inliner, imgWarp
    else:
        return 0, 0, good, img1
        #print('%d matches found, not enough for homography estimation' % len(p1))

if __name__ == '__main__':
    print(__doc__)

    import sys, getopt
    opts, args = getopt.getopt(sys.argv[1:], '', ['feature='])
    opts = dict(opts)
    feature_name = opts.get('--feature', 'orb')
    fn1, fn2 = args
    #img1 = remove_bg(fn1)
    #img2 = remove_bg(fn2)
    src = cv2.imread(fn1)
    dst = cv2.imread(fn2)
    detector, matcher = init_feature(feature_name)

    if src is None:
        print('Failed to load fn1:', fn1)
        sys.exit(1)

    if dst is None:
        print('Failed to load fn2:', fn2)
        sys.exit(1)

    if detector is None:
        print('unknown feature:', feature_name)
        sys.exit(1)
    
    srcImg, srcLoc = seg_img(src, fn1)
    dstImg, dstLoc = seg_img(dst, fn2)
    print("total sub images:" + str(len(srcLoc)) + " " + str(len(dstLoc)))
    print('using', feature_name)
    si = 0
    for i in srcLoc:
        img1 = srcImg[i[1]:i[1] + i[3], i[0]:i[0] + i[2]] 
        if i[3] < 100 or i[2] < 100:
            print("src too small")
            continue
        cv2.normalize(img1, img1, 0, 255, cv2.NORM_MINMAX)
        di = 0
        for j in dstLoc:
            if j[3] < 100 or j[2] < 100:
                print("dst too small")
                continue
            img2 = dstImg[j[1]:j[1] + j[3], j[0]:j[0] + j[2]]   
            cv2.normalize(img2, img2, 0, 255, cv2.NORM_MINMAX)
            
            kp1, desc1 = detector.detectAndCompute(img1, None)
            #cv2.imwrite('kp-' + fn1, cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
            kp2, desc2 = detector.detectAndCompute(img2, None)
            #cv2.imwrite('kp-' + fn2, cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS))
            di = di + 1
            try:
                matched, matches, good, imgWarp = match_and_draw(kp1, desc1, kp2, desc2, img1, img2, si, di)
            except:
                matched = 0
                matches = 0
                good = []
                raw = []
                imgWarp = img1

            print("src %d features %d, dst %d features %d" % (si, len(desc1), di, len(desc2)))
            min_features = len(desc1) #(len(desc1) + len(desc2))/2
            matched_ratio = matches / min_features
            
            if matched > 80 and matched_ratio > 0.5:    
                #img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
                img3 = cv2.hconcat([imgWarp, img2])
                # Show the image
                print('%d%% matched, good matches %s, matched_ratio %s' % (matched, matches, matched_ratio))
                cv2.imwrite("matched-" + str(int(matched)) + "-" + str(si) + "-" + str(di) + ".jpg", img3)
            else:
                if matches > 0:
                    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
                    
                    print('%d%% matched, good matches %s matched_ratio %s' % (matched, matches, matched_ratio))
                    cv2.imwrite("matched-bad-" + str(int(matched))+ "-" +  str(si) + "-" + str(di) + ".jpg", img3)
                    #img4 = cv2.hconcat([imgWarp, img2])
                    #cv2.imwrite("matched-warp-" + str(int(matched))+ "-" +  str(si) + "-" + str(di) + ".jpg", img4)

        si = si + 1
            # cv2.waitKey()
    # cv2.destroyAllWindows()
