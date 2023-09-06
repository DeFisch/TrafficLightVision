import prepare_data as data
import detect as det
import matplotlib.pyplot as plt
import cv2 as cv

# control params
show_filter_layers = True # if wanna see how color is filtered, set to True
save_result = True # if results are to be saved instead of shown, set to True
SAVE_PATH = '../results/'

# load images
images = data.images
cnt = 1
for img in images:
    # get results from the computation
    [reds, yellows, greens] = det.detect_img(img,show_filter_layers)
    circles = [reds,yellows,greens]
    colors = ['RED','YELLOW','GREEN']
    # define color for labels
    color_val = [(0,0,150),(0,150,150),(0,150,0)]

    # visualize data
    img_cpy = img.copy()
    height,width,_ = img_cpy.shape

    # mark the image
    for i in range(3):
        if circles[i] is not None:
            j = 0
            for circle in circles[i][0, :]:
                cv.rectangle(img_cpy,(int(circle[0]-circle[2]),int(circle[1]-circle[2])),(int(circle[0]+circle[2]),int(circle[1]+circle[2])),color_val[i],thickness=max(int(5*(width/1600)),1))
                text_size, _ = cv.getTextSize(colors[i], cv.FONT_HERSHEY_SIMPLEX, (width/1600), max(int(3*(width/1600)),1))
                w,h = text_size
                cv.rectangle(img_cpy,(int(circle[0]-circle[2]+w),int(circle[1]-circle[2]-h)),(int(circle[0]-circle[2]),int(circle[1]-circle[2])),color_val[i],thickness=-1)
                cv.putText(img_cpy,colors[i],(int(circle[0]-circle[2]),int(circle[1]-circle[2])),cv.FONT_HERSHEY_SIMPLEX,fontScale=(width/1600),color=(255,255,255),thickness=max(int(3*(width/1600)),1))
                
                j += 1

    # show/save marked image
    if save_result:
        cv.imwrite(SAVE_PATH+f'img_{cnt}.jpg', img_cpy)
    else:         
        cv.imshow('display',img_cpy)
        cv.waitKey(0)
    cnt += 1

print('Task Finished.')

