'''
For a given data source, create the image, annotation and 
categories info
'''

from cocoDataStructure.images import getImageInfo
from cocoDataStructure.annotations import getAnnotationInfo
from cocoDataStructure.categories import getCategoriesInfo
from cocoDataStructure.utilities import *

import json
from glob import glob
from itertools import repeat
import os

if __name__ == "__main__":

    srcs = ["/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/",
    "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Fish4Knowledge/",
    "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/openimages/",
    "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/QUT/"]

    for src in srcs:
        if os.path.isdir(src):
            imgDict = associateImageID(src)
            imageInfo = getImageInfo(src)
            annotationInfo = getAnnotationInfo(src)