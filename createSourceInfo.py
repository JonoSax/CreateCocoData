'''
For a given data source, create the image, annotation and 
categories info
'''

from cocoDataStructure.images import getImageInfo
from cocoDataStructure.annotations import getAnnotationInfo
from cocoDataStructure.categories import getCategoriesInfo
from cocoDataStructure.utilities import *
from multiprocessing import Pool

import json
from glob import glob
from itertools import repeat
import os

if __name__ == "__main__":

    cpuNo = 4
    
    wd = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/CocoData/"
    wd = "/Volumes/USB/data/CocoData/"

    srcs = [wd + "Ulucan/",
    wd + "Fish4Knowledge/",
    wd + "openimages/",
    wd + "QUT/"]

    # srcs = ["/Volumes/WorkStorage/BoxFish/dataStore/Aruco+Net/net_day_shade_pool/"]
    # srcs = ["/Volumes/WorkStorage/BoxFish/dataStore/netData/foregrounds/mod/"]

    '''
    for s in srcs:
        associateImageID(s)
        getImageInfo(s)
        getAnnotationInfo(s)
    '''
    with Pool(cpuNo) as pool:
        pool.map(associateImageID, srcs)
        pool.map(getImageInfo, srcs)
        pool.map(getAnnotationInfo, srcs)