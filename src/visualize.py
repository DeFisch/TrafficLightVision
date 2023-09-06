import prepare_data as data
import detect as det
import matplotlib.pyplot as plt
import cv2 as cv

# load images
images = data.images
sub_image = []
sub_image.append(images[11])
for img in images:
    # get results from the computation
    show_filter_layers = False # if wanna see how color is filtered, set to True
    [reds, yellows, greens] = det.detect_img(img,show_filter_layers)
    circles = [reds,yellows,greens]
    colors = ['RED','YELLOW','GREEN']
    # define color for labels
    color_val = [(61,61,255),(0,206,229),(0,174,17)]

    # visualize data
    img_cpy = img.copy()
    height,width,_ = img_cpy.shape
    for i in range(3):
        if circles[i] is not None:
            j = 0
            for circle in circles[i][0, :]:
                cv.circle(img_cpy,(int(circle[0]),int(circle[1])),int(circle[2]),color_val[i],3)
                cv.putText(img_cpy,colors[i],(int(circle[0]-circle[2]),int(circle[1]-circle[2])),cv.FONT_HERSHEY_SIMPLEX,fontScale=(width/1600),color=color_val[i],thickness=3)
                j += 1
                
    cv.imshow('display',img_cpy)
    cv.waitKey(0)


