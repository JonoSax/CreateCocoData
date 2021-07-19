'''
For a given data source, create the image, annotation and 
categories info
'''

from images import getImageInfo
from annotations import getAnnotationInfo
from categories import getCategoriesInfo
from utilities import *

import json
from glob import glob
from itertools import repeat

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/"

    imgDict = associateImageID(src)

    categoryInfo = getCategoriesInfo(src)
    imageInfo = getImageInfo(src)
    annotationInfo = getAnnotationInfo(src)