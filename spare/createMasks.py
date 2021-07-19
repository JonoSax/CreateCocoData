'''
This script creaates masks from either segmentation or 
bounding box information.
'''

import numpy as np
from glob import glob
import cv2
import json
from utilities import *

def getBBoxAnnotation(src):

    '''
    From a list of images and the mask information, create the 
    bounding box mask
    '''

    maskSrc = glob(src + "*.json")[0]

    maskInfo = json.load(open(maskSrc))

    annotations = maskInfo['annotations']
    images = maskInfo['images']

    imageBBox = createIDDict(annotations, "image_id", "bbox")
    imagePath = createIDDict(images, "id", "file_name", src)

    for i in imageBBox:
        img = cv2.imread(imagePath[i][0])
        bboxes = imageBBox[i]

        for bbox in bboxes:
            bbox = np.array(bbox)
            imgMod = cv2.rectangle(img, bbox[0:2], bbox[0:2]+bbox[2:4], (0,0,255), 2)

        cv2.imshow("img", imgMod)
        cv2.waitKey(0)

    return

def getSegmentMask(imgs, maskSrc):

    '''
    From a list of images and the mask information, create the 
    bounding box mask
    '''



    return

if __name__ == '__main__':

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Aquarium/train/"
    getBBoxAnnotation(src)