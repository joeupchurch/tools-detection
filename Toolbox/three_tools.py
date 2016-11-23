import numpy as np
import cv2
from matplotlib import pyplot as plt

images = ['tools1.jpg']

for img_str in images:

    # Read image
    img = cv2.imread(img_str)
    
    # Gather length, width data on image
    length,width,temp = img.shape
    
    # Convert image to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    # Equalize contrast
    equ = cv2.equalizeHist(gray)
    
    # Threshold on brightness
    ret, thresh = cv2.threshold(gray,127,255,cv2.THRESH_BINARY)
    
    # Perform morphological opening
    kernel = np.ones((width/40,width/40),np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    # Find outermost contour
    im2,contours,hierarchy = cv2.findContours(opening, 1, 2)
    cnt = contours[-1]
    
    # Approximate contour to remove concavity
    epsilon = 0.1*cv2.arcLength(cnt,True)
    approx = cv2.approxPolyDP(cnt,epsilon,True)
    
    # Gather outermost contour points
    leftmost = tuple(approx[approx[:,:,0].argmin()][0])
    rightmost = tuple(approx[approx[:,:,0].argmax()][0])
    topmost = tuple(approx[approx[:,:,1].argmin()][0])
    bottommost = tuple(approx[approx[:,:,1].argmax()][0])
    
    # Determine new image length and width
    transwidth = rightmost[0]-leftmost[0]-1
    translength = bottommost[1]-topmost[1]-1
    
    # Display outer contour
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.drawContours(img,[approx],0,(0,0,255),2)
    cv2.imshow('Image',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    # Create transform 
    pts1 = np.float32([approx[0],approx[3],approx[1],approx[2]])
    pts2 = np.float32([[0,0],[transwidth,0],[0,translength],[transwidth,translength]])
    
    # Transform Perspective based on limits of drawer
    M = cv2.getPerspectiveTransform(pts1,pts2)
    dst = cv2.warpPerspective(img,M,(transwidth,translength))
    
    # Display transformed image
    cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
    cv2.imshow('Image',dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Convert transformed image to grayscale
    graydst = cv2.cvtColor(dst,cv2.COLOR_BGR2GRAY)

    # Threshold to remove brightest values 
    ret,threshdst1 = cv2.threshold(graydst,245,255,cv2.THRESH_TOZERO_INV)
    res2 = np.hstack((graydst,threshdst1))

    # Perform CLAHE equalization
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl1 = clahe.apply(threshdst1)
    res = np.hstack((threshdst1,cl1))

    # Threshold to remove background
    retdst, threshdst2 = cv2.threshold(cl1,145,255,cv2.THRESH_BINARY_INV)

    # Perform morphological opening
    kerneldst = np.ones((3,3),np.uint8)
    opendst = cv2.morphologyEx(threshdst2, cv2.MORPH_CLOSE, kerneldst,iterations = 10)

##circles = cv2.HoughCircles(opendst,cv2.HOUGH_GRADIENT,1,20,
##                    param1=30,param2=30,minRadius=0,maxRadius=transwidth)

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',opendst)
cv2.waitKey(0)
cv2.destroyAllWindows()

cutlines = np.ones((translength,transwidth,3),np.uint8)*255

im2dst, contoursdst, hierarchy = cv2.findContours(opendst,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
borderperc = 0.1
areaperc = .0005
quarterdiam = 0.955
dpi = 300
for i in range(hierarchy.shape[1]):
    cnt = contoursdst[i]
    M = cv2.moments(cnt)
    area = int(M['m00'])
    if area>int(translength*transwidth*areaperc):
        cx = int(M['m10']/M['m00'])
        cy = int(M['m01']/M['m00'])
        if (cx>int(transwidth*borderperc) and cx<int(transwidth*(1-borderperc)) and
            cy>int(translength*borderperc) and cy<int(translength*(1-borderperc))):
            
            (x,y),radius = cv2.minEnclosingCircle(cnt)
            center = (int(x),int(y))
            circarea = int(3.14*(radius*radius))
            if circarea>int(area*.9) and circarea<int(area*1.1):
                cv2.circle(dst,center,2,(255,0,0),2)
                cv2.circle(dst,center,int(radius),(0,255,0),2)
                scalefactor = radius*2/quarterdiam
            else:
                hull = cv2.convexHull(cnt)
                if hull.shape[0]>1:
                    for j in range(hull.shape[0]):
                        if j==hull.shape[0]-1:
                            start = tuple(hull[j][0])
                            end = tuple(hull[1][0])
                        else:
                            start = tuple(hull[j][0])
                            end = tuple(hull[j+1][0])
                        cv2.line(dst,start,end,[0,0,255],3)
                        cv2.line(cutlines,start,end,[0,0,255],3)


cv2.line(dst,(0,0),(0,translength),[255,0,0],3)
cv2.line(dst,(0,translength),(transwidth,translength),[255,0,0],3)
cv2.line(dst,(transwidth,translength),(transwidth,0),[255,0,0],3)
cv2.line(dst,(transwidth,0),(0,0),[255,0,0],3)

cv2.line(cutlines,(0,0),(0,translength),[255,0,0],3)
cv2.line(cutlines,(0,translength),(transwidth,translength),[255,0,0],3)
cv2.line(cutlines,(transwidth,translength),(transwidth,0),[255,0,0],3)
cv2.line(cutlines,(transwidth,0),(0,0),[255,0,0],3)

scaledst = cv2.resize(dst,(int(dpi/scalefactor*transwidth),int(dpi/scalefactor*translength)),
                      interpolation = cv2.INTER_LINEAR)

scalecutlines = cv2.resize(cutlines,(int(dpi/scalefactor*transwidth),int(dpi/scalefactor*translength)),
                      interpolation = cv2.INTER_LINEAR)

scaleorig = cv2.resize(img,(transwidth,translength),interpolation = cv2.INTER_LINEAR)


##circles = np.uint16(np.around(circles))
##for i in circles[0,:]:
##    # draw the outer circle
##    cv2.circle(dst,(i[0],i[1]),i[2],(255,0,0),2)
##    # draw the center of the circle
##    cv2.circle(dst,(i[0],i[1]),2,(0,0,255),3)
        
##print im2dst
##print hierarchy

##cntdst = contoursdst[-1]
##
##epsilon = 0.1*cv2.arcLength(cntdst,True)
##approx = cv2.approxPolyDP(cntdst,epsilon,True)
##
##cv2.drawContours(dst, approx, -1, (255,0,0), 3)
##

##cntdst = contoursdst[4]
##hull = cv2.convexHull(contoursdst)
##cv2.drawContours(dst, cv2.convexHull(contoursdst[-1]), -1, (255,0,0), 3)



##print contoursdst

endprod = np.hstack((scaleorig,dst))

cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
cv2.imshow('Image',endprod)
cv2.waitKey(0)
cv2.destroyAllWindows()

##cv2.imwrite('Tools1Processed.jpg',endprod)


##cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
##cv2.imshow('Image',scaledst)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##cv2.imwrite('Tools1Cutlines.png',scalecutlines)

##im2dst,contoursdst,hierarchydst = cv2.findContours(threshdst, 1, 2)
##cntdst = contoursdst[-1]
##
##cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
##cv2.imshow('Image',dst)
##cv2.waitKey(0)
##cv2.destroyAllWindows()
##
##hulldst = cv2.convexHull(cntdst,returnPoints = False)
##defectsdst = cv2.convexityDefects(cntdst,hulldst)
##
##for i in range(defectsdst.shape[0]):
##    s,e,f,d = defectsdst[i,0]
##    start = tuple(cntdst[s][0])
##    end = tuple(cntdst[e][0])
##    far = tuple(cntdst[f][0])
##    cv2.line(dst,start,end,[0,255,0],2)
##    cv2.circle(dst,far,5,[0,0,255],-1)
##
##cv2.namedWindow('Image',cv2.WINDOW_NORMAL)
##cv2.imshow('img',dst)
##cv2.waitKey(0)
##cv2.destroyAllWindows()

