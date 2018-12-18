import cv2

import numpy as np 

import math

import random

from task1 import sift_features

np.random.seed(sum([ord(c) for c in 'srivenka']))

def drawlines(img1,img2,lines,pts1,pts2):   
    r,c = img1.shape
    img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for r,pt1,pt2 in zip(lines,pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = map(int, [0, -r[2]/r[1] ])
        x1,y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0,y0), (x1,y1), color,1)
        img1 = cv2.circle(img1,tuple(pt1),5,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),5,color,-1)
    return img1,img2


def knnmatchandfundamentalmatrix(image1,image2,keypoints_tsucuba_left,descriptors_tsucuba_left,keypoints_tsucuba_right, descriptors_tsucuba_right):
    matcher = cv2.BFMatcher()
    matches = matcher.knnMatch(descriptors_tsucuba_left, descriptors_tsucuba_right, k=2)

    good_match = []
    good_match_new = []
    good_match_random = []
    pts1 = []
    pts2 = []
    new_pts1 = []
    new_pts2 = []
    for i,j in matches:
        if i.distance < 0.75 * j.distance:
            good_match.append([i])
            good_match_new.append(i)
            pts2.append(keypoints_tsucuba_right[i.trainIdx].pt)
            pts1.append(keypoints_tsucuba_left[i.queryIdx].pt)

    knn_image = cv2.drawMatchesKnn(image1, keypoints_tsucuba_left,image2,keypoints_tsucuba_right,good_match,None)
    cv2.imwrite('task2_matches_knn.jpg', knn_image)

    

        
    pts1 = np.int32(pts1)
    pts2 = np.int32(pts2)
    F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC)
    pts1 = pts1[mask.ravel()==1]
    pts2 = pts2[mask.ravel()==1]
    print(F)    
    for i in range(10):
        rand = random.randint(0,len(pts1)-1)
        new_pts1.append(pts1[rand])
        new_pts2.append(pts2[rand])

    new_pts1 = np.int32(new_pts1)
    new_pts2 = np.int32(new_pts2)
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(new_pts2.reshape(-1,1,2), 2,F)
    lines1 = lines1.reshape(-1,3)
    img5,img6 = drawlines(image1,image2,lines1,new_pts1,new_pts2)

# Find epilines corresponding to points in left image (first image) and
# drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(new_pts1.reshape(-1,1,2), 1,F)
    lines2 = lines2.reshape(-1,3)
    img3,img4 = drawlines(image2,image1,lines2,new_pts2,new_pts1)


    cv2.imwrite('task2_epi_right.jpg',img3)
    cv2.imwrite('task2_epi_left.jpg',img5)

    stereo = cv2.StereoBM_create(numDisparities=96, blockSize=31)
    disparity = stereo.compute(image1,image2)
    cv2.imwrite('task2_disparity.jpg',disparity)

    return


if __name__ == "__main__":

    tsucuba_left = cv2.imread('tsucuba_left.png')
    tsucuba_right = cv2.imread('tsucuba_right.png')

    tsucuba_left_grey = cv2.imread('tsucuba_left.png',0)
    tsucuba_right_grey = cv2.imread('tsucuba_right.png',0)


    keypoints_tsucuba_left , descriptors_tsucuba_left = sift_features(tsucuba_left_grey,1,2)
    keypoints_tsucuba_right , descriptors_tsucuba_right = sift_features(tsucuba_right_grey,2,2)

    knnmatchandfundamentalmatrix(tsucuba_left_grey,tsucuba_right_grey,keypoints_tsucuba_left,descriptors_tsucuba_left,keypoints_tsucuba_right, descriptors_tsucuba_right)