'''
Get the coco image informatino for a given data source

id
license
file_name
height
width
date_captured

'''

from utilities import printProgressBar
import numpy as np
from glob import glob
import cv2
import json

def getImageInfo(src):

    '''
    Populate the image info for a given data source
    '''

    idDict = json.load(open(src + "imgDict.json"))

    imgs = sorted(glob(src + "masks/**/*"))

    imgsInfo = []

    for n, i in enumerate(imgs):
        imgName = i.split("/")[-1]
        printProgressBar(n, len(imgs) - 1, prefix = 'ImageInfo:', suffix = 'Complete', length = 15)
        img = cv2.imread(i)
        imgDict = {}
        imgDict['file_name'] = i
        imgDict['id'] = idDict[imgName]
        imgDict['height'] = img.shape[0]
        imgDict['width'] = img.shape[1]
        imgDict['date_captured'] = ""                # date captured not critical
        imgsInfo.append(imgDict)

    # save the dictionary as a json file in the src
    json.dump(imgsInfo, open(src + "images.json", 'w'))

    return(imgsInfo)

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Fish4Knowledge/"

    getImageInfo(src)

    pass