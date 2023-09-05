import prepare_data as data
import detect as det
import matplotlib.pyplot as plt
import cv2 as cv

# prepare and compute circles
images = data.images
for img in images:
    [reds, yellows, greens] = det.detect_img(img)
    circles = [reds,yellows,greens]
    colors = ['RED','YELLOW','GREEN']

    # visualize data
    img_cpy = img.copy()
    height,width,_ = img_cpy.shape
    for i in range(3):
        if circles[i] is not None:
            j = 0
            for circle in circles[i][0, :]:
                cv.circle(img_cpy,(int(circle[0]),int(circle[1])),int(circle[2]),(0,255,0),2)
                cv.putText(img_cpy,colors[i],(int(circle[0]),int(circle[1])),cv.FONT_HERSHEY_SIMPLEX,fontScale=(width/1600),color=(0,0,255),thickness=3)
                j += 1
                
    cv.imshow('display',img_cpy)
    cv.waitKey(0)
