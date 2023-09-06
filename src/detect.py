import cv2 as cv
import numpy as np
import math

def detect_img(img,show_layers): # show_layers: show filtered color layers if True

    # get image dimensions
    height,width,_ = img.shape

    # convert target img to hsv color space
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # apply bilateral filter to smooth out some textures
    hsv = cv.bilateralFilter(hsv, 9,75,75 )
    
    # separate red green yellow color channels and filter out target color
    lower_red1 = np.array([0,30,200])
    upper_red1 = np.array([10,255,255])
    lower_red2 = np.array([170,30,200])
    upper_red2 = np.array([180,255,255])
    lower_green = np.array([50,60,200])
    upper_green = np.array([95,255,255])
    lower_yellow = np.array([22,115,210])
    upper_yellow = np.array([33,255,255])
    maskr1 = cv.inRange(hsv, lower_red1, upper_red1)
    maskr2 = cv.inRange(hsv, lower_red2, upper_red2)
    maskg = cv.inRange(hsv, lower_green, upper_green)
    masky = cv.inRange(hsv, lower_yellow, upper_yellow)
    maskr = cv.add(maskr1, maskr2)

    # apply gaussian blur to smooth out edges
    masky = cv.GaussianBlur( masky,(5,5),0 )
    maskr = cv.GaussianBlur( maskr,(5,5),0 )
    maskg = cv.GaussianBlur( maskg,(5,5),0 )

    # show filtered color layers
    if show_layers:
        cv.imshow('red',maskr)
        cv.waitKey(0)  
        cv.imshow('yellow',masky)
        cv.waitKey(0)  
        cv.imshow('green',maskg)
        cv.waitKey(0)

    # compress image if it's too large
    dp = 1
    if width*height > 1920*1080:
        dp = 1.5
    min_dist = 50

    # hough circle detection with limits on circle size, distance, and perfectness
    red_circles = cv.HoughCircles(maskr, cv.HOUGH_GRADIENT_ALT, dp, min_dist,
                               param1=30, param2=.5, minRadius=0, maxRadius=1000)
    green_circles = cv.HoughCircles(maskg, cv.HOUGH_GRADIENT_ALT, dp, min_dist,
                               param1=30, param2=.75, minRadius=0, maxRadius=1000)
    yellow_circles = cv.HoughCircles(masky, cv.HOUGH_GRADIENT_ALT, dp, min_dist,
                               param1=30, param2=.5, minRadius=0, maxRadius=1000)
    
    circles = [red_circles,yellow_circles,green_circles]
    colors = ['red', 'yellow', 'green']
    
    for i in range(3):
        del_idx = []
        if circles[i] is not None:
            j = 0
            for circle in circles[i][0, :]:
                x,y = int(circle[0]),int(circle[1])
                rad = math.floor(circle[2]/(math.sqrt(2)))
                # crop out traffic light candidates
                cropped_bgr = img[y-rad:y+rad,x-rad:x+rad]
                bgr_var = np.var(cropped_bgr[:,:,0])+np.var(cropped_bgr[:,:,1])+np.var(cropped_bgr[:,:,2])
                bgr_var = int(bgr_var)
                # calculate regional mean hue, saturation, and value
                avg_hue,avg_sat,avg_val=rgb_to_hsv(int(np.mean(cropped_bgr[:,:,2])),int(np.mean(cropped_bgr[:,:,1])),int(np.mean(cropped_bgr[:,:,0])))
                # remove candidate if mean hue, saturation, and value does not match
                if (i == 0 and (18 < avg_hue < 160 or avg_sat < 120 or avg_val < 170)) or (i == 1 and (not(18 < avg_hue < 33) or not(200 < avg_val < 255))) or (i == 2 and (not(50 < avg_hue < 95))):
                    del_idx.append(j)
                # remove candidates on the lower 25% of the screen
                elif circle[1] > 0.75*height:
                    del_idx.append(j)
                else:
                    print(f'{colors[i]} traffic light at: {circle[0]}, {circle[1]}')
                j+=1
        # remove all the non-candidates
        for idx in reversed(del_idx):
            circles[i] = np.delete(circles[i],idx,1)

    return circles


# code from https://www.w3resource.com/python-exercises/math/python-math-exercise-77.php
# simple conversion from rgb to hsv
def rgb_to_hsv(r, g, b):
    r, g, b = r/255.0, g/255.0, b/255.0
    mx = max(r, g, b)
    mn = min(r, g, b)
    df = mx-mn
    if mx == mn:
        h = 0
    elif mx == r:
        h = (60 * ((g-b)/df) + 360) % 360
    elif mx == g:
        h = (60 * ((b-r)/df) + 120) % 360
    elif mx == b:
        h = (60 * ((r-g)/df) + 240) % 360
    if mx == 0:
        s = 0
    else:
        s = (df/mx)*100
    v = mx*100
    return h/2, s*2.25, v*2.25