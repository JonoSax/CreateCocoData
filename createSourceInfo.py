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

    cpuNo = 1
    
    wd = "/media/boxfish/USB/data/CocoData/"
    wd = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/CocoData/"
    wd = "/Volumes/USB/data/CocoData/"

    srcs = [wd + "Ulucan/",
    wd + "Fish4Knowledge/",
    wd + "openimages/",
    wd + "QUT/", 
    wd + "NorFisk_v1.0/",
    wd + "BrackishWaterImages/"]

    # srcs = ["/Volumes/WorkStorage/BoxFish/dataStore/Aruco+Net/net_day_shade_pool/"]
    # srcs = ["/Volumes/WorkStorage/BoxFish/dataStore/netData/foregrounds/mod/"]

    if cpuNo == 1:
        for s in srcs[-1:]:
            associateImageID(s)
            getImageInfo(s)
            getAnnotationInfo(s)
    else:
        # NOTE parallelise the individual functions
        with Pool(cpuNo) as pool:
            pool.map(associateImageID, srcs)
            pool.map(getImageInfo, srcs, True)
            # pool.map(getAnnotationInfo, srcs)