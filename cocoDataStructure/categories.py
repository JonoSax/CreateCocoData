'''
Gets the coco category information for a given source


id
name
supercategory

'''

import numpy as np
from glob import glob
import cv2
from cocoDataStructure.utilities import *
import json
import os

def getCategoriesInfo(src):

    '''
    Populate the category info for a given data source
    '''

    # from all the image files, get the uniqe categories
    imgsrc = src + "*/images/"
    dirs = np.unique([d.split("/")[-1] for d in sorted(glob(imgsrc + "*"))])

    categoryInfo = []

    for n, d in enumerate(dirs):
        categoryDict = {}
        categoryDict["id"] = n + 1
        categoryDict["name"] = d
        categoryDict["supercategory"] = ""
        categoryInfo.append(categoryDict)

    json.dump(categoryInfo, open(src + "categories.json", "w"))

    return(categoryInfo)

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/"

    categoryInfo = getCategoriesInfo(src)
