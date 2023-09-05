import numpy as np
import cv2 as cv

num_data = 14
index = np.linspace(1,num_data,num=num_data,dtype=int)
images = []

for i in index:
    images.append(cv.imread(f'./data/{i}.jpg'))