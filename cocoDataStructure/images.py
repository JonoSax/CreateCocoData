'''
Get the coco image informatino for a given data source

id
license
file_name
height
width
date_captured

'''

if __name__ == "__main__":
    from utilities import *
else:
    from cocoDataStructure.utilities import *
import multiprocessing
from multiprocessing import Pool
import numpy as np
from glob import glob
import cv2
import json
from itertools import repeat


def getImgInfo(i, relative_path, idDict, q = None):

    imgName = i.split("/")[-1]
    # printProgressBar(n, len(imgs) - 1, prefix = f'getImageInfo:{imgName}', suffix = 'Complete', length = 15)
    img = cv2.imread(i)
    imgDict = {}
    if relative_path:
        imgDict["file_name"] = "/".join(i.split("/")[-4:])      # images are 4 levels deep in the directory
    else:
        imgDict['file_name'] = i
    imgDict['id'] = idDict[imgName]
    imgDict['height'] = img.shape[0]
    imgDict['width'] = img.shape[1]
    imgDict['date_captured'] = ""                # date captured not critical
    
    return(imgDict)

def getImageInfo(src, relative_path=False, cpuNo = 4):

    '''
    Populate the image info for a given data source
    '''

    print(f"Analysing {src.split('/')[-2]} images")

    idDict = json.load(open(src + "imgDict.json"))

    imgs = getAllImages(src)

    if cpuNo > 1:
        with Pool(cpuNo) as p:
            imgsInfo = p.starmap(getImgInfo, zip(imgs, repeat(relative_path), repeat(idDict), ))
    else:
        imgsInfo = []
        for n, i in enumerate(imgs):
            imgDict = getImgInfo(i, relative_path, idDict)
            imgsInfo.append(imgDict)


    # save the dictionary as a json file in the src
    if imgsInfo != []:
        json.dump(imgsInfo, open(src + "images.json", "w"))

    print(f"    Finished {src.split('/')[-2]}")

    return(imgsInfo)

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Fish4Knowledge/"
    src = "/media/boxfish/USB/data/CocoData/Ulucan/"
    getImageInfo(src)
