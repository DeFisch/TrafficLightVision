
import cv2 as cv
import os

images = []
folder = '../data/'
for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    if file_path.endswith('.png'):
        try:
            images.append(cv.imread(file_path,cv.IMREAD_UNCHANGED))
        except Exception as e:
            print(f'Failed to read {file_path}: {e}')

    