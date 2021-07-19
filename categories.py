'''
Gets the coco category information for a given source


id
name
supercategory

'''

import numpy as np
from glob import glob
import cv2
from utilities import *
import json
import os

from utilities import *

def getCategoriesInfo(src):

    '''
    Populate the category info for a given data source
    '''

    imgsrc = src + "images/"
    dirs = [d.split(imgsrc)[-1] for d in sorted(glob(imgsrc + "*"))]

    categoryInfo = []

    for n, d in enumerate(dirs):
        categoryDict = {}
        categoryDict["id"] = n
        categoryDict["name"] = d
        categoryDict["supercategory"] = ""
        categoryInfo.append(categoryDict)

    json.dump(categoryInfo, open(src + "categories.json", "w"))

    return(categoryInfo)

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Fish4Knowledge/"

    categoryInfo = getCategoriesInfo(src)
