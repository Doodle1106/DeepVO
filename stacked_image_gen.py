# coding=UTF-8
import os
import cv2
import numpy as np

data_dir = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/data/'
stacked_dir = '/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/stacked/'
f = os.listdir(data_dir)
f.sort()

prev = cv2.imread(data_dir+f[0])
print (len(f))
print (data_dir+f[0])

# a = cv2.imread('/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/data/1403715523912143104.png')
# b = cv2.imread('/home/shr/software/softwarebackup/EUROC/V1_02_medium/mav0/cam0/data/1403715602762142976.png')
# combined = np.concatenate((a,b), axis=0)
# '''a在上， b在下'''

j = 1
for i in f[1:]:
    img = data_dir + i
    print (img)
    img = cv2.imread(img)
    combined = np.concatenate((img, prev), axis=0)
    print (np.shape(combined))
    prev = img
    # cv2.imshow("image", combined)
    print (stacked_dir + i)

    #use current image's name as current saved file's name
    cv2.imwrite(stacked_dir+i, combined)
    j += 1
    # cv2.waitKey(1000)

