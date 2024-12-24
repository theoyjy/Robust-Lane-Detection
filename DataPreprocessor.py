

import os
from PIL import Image
import cv2
import numpy as np


def readTxt(file_path):
    img_list = []
    # print file full path
    print('file full path:', os.path.abspath(file_path))
    with open(file_path, 'r') as file_to_read:
        while True:
            lines = file_to_read.readline()
            if not lines:
                break
            item = lines.strip().split()
            img_list.append(item)
    file_to_read.close()
    return img_list

# read CUList/val.txt to get the image path list 
val_path = 'CUList/test.txt'
val_list = readTxt(val_path)

# for each image, 
new_val_list = []
counter = 0
tmpList = []
curPath = os.path.abspath('.')
curPath = curPath.replace("\\", "/")
mainFolder = ""
subfolder = ""
for i in range(len(val_list)):
    img_path_list = val_list[i]
    tmpMainFolder = img_path_list[0].split('/')[-3]
    newSubfolder = img_path_list[0].split('/')[-2]

    if mainFolder != tmpMainFolder:
        mainFolder = tmpMainFolder
        subfolder = newSubfolder
        tmpList = []
        counter = 0
        
    # new subfolder, reset the counter and tmpList
    if subfolder != newSubfolder:
        tmpList = []
        counter = 0
        subfolder = newSubfolder

    # replace the original path's frist part with driver_23_resized
    resized_path = curPath + img_path_list[0].replace(mainFolder, mainFolder + '_resized')
    # append the new path to tmpList
    tmpList.append(resized_path)
    counter += 1
    
    # resize to 256*128 and
    if not os.path.exists(resized_path):
        data = Image.open(curPath + img_path_list[0])
        data = data.resize((256, 128), Image.LANCZOS)
        os.makedirs(os.path.dirname(resized_path), exist_ok=True)
        data.save(resized_path)


    if counter == 5:
        # read the original image since the label is based on the original image
        mask_path = resized_path.replace('jpg', 'truth.jpg')
        # if not os.path.exists(mask_path):

        data = Image.open(curPath + img_path_list[0])
        size = data.size
        data.close()

        # read label txt by replacing ends with .lines.txt
        label_path = img_path_list[0].replace('jpg', 'lines.txt')
        label = Image.new('L', size, 0)
        
        with open(curPath + label_path, 'r') as file_to_read:
            while True:
                lines = file_to_read.readline()
                if not lines:
                    break
                item = lines.strip().split()
                # draw the lines on the label image
                for j in range(0, len(item) - 2, 2):
                    x1, y1 = int(float(item[j])), int(float(item[j + 1]))
                    x2, y2 = int(float(item[j + 2])), int(float(item[j + 3]))
                    # draw the lines on the label image
                    label = cv2.line(np.array(label), (x1, y1), (x2, y2), 255, 4)
                label = Image.fromarray(label)

        label = label.resize((256, 128), Image.NEAREST)
        label.save(mask_path, quality=100)

        tmpList.append(mask_path)
        tmpList_str = ' '.join(tmpList)  # Convert tmpList to a string
        new_val_list.append(tmpList_str)
        counter = 0
        tmpList = []

# write the new_val_list to a new file
new_val_path = '/CUList/test_resized.txt'
with open(curPath + new_val_path, 'w') as file:
    for item in new_val_list:
        file.write("%s\n" % item)