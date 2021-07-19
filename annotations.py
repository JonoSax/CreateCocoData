'''
Gets the coco annotation information for a given source

id
image_id
category_id
bbox
area
segmentation
iscrowd

'''

import numpy as np
from glob import glob
import cv2
from utilities import *
import json

def getAnnotations(img):

    '''
    Gets all the mask info for the annotaiton
    '''

    dims = getDims(img)
    pixels, segment = getSegmentation(img)
    bbox = getBoundingBox(pixels)
    area = getArea(pixels)

    segment = processCoco(segment)
    bbox = processCoco(bbox)

    return(pixels, segment, area, bbox, dims)

def getDims(img):

    '''
    Get the image dimensions
    '''

    x, y, _ = img.shape

    return(x, y)

def getSegmentation(img):

    '''
    Get the list of points which make up the segmentation mask
    '''

    # count all the non-zero pixel positions as the mask
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # raw position info
    pixels = np.where(imgGray != 0)


    border = []
    # get the top of the image (left to right)
    for i in range(imgGray.shape[0]):
        pos = np.where((imgGray[i, :])==255)[0]
        if len(pos) > 0:
            border.append(max(pos))
            border.append(i)

    # get the bottom of the image (right to left)
    for i in range(imgGray.shape[0]-1, -1, -1):
        pos = np.where((imgGray[i, :])==255)[0]
        if len(pos) > 0:
            border.append(min(pos))
            border.append(i)


    # formatted for the segment info
    segment = str([border])

    return(pixels, segment)

def getBoundingBox(pixel):

    '''
    Get the bounding box of an images segmentation mask
    (topLeft_X, topLeft_Y, width, height)
    '''

    topL_y, topL_x = np.min(pixel, axis = 1)
    botR_y, botR_x = np.max(pixel, axis = 1)

    width = botR_x - topL_x
    height = botR_y - topL_y

    return(str([topL_x, topL_y, width, height]))

def getArea(pixel):

    '''
    Get the number of pixels within an images segmentation mask
    '''

    posCount = len(pixel[0])

    return(posCount)

def processCoco(cocoInfo):

    '''
    To get the coco structure correct some processing is required...
    '''

    # remove string inserts
    cocoInfoNew = cocoInfo.replace('[', '').replace(']', '').replace(" ", "")
    # to put back together: [int(b) for b in a[0]["segmentation"].split(",")]

    return(cocoInfoNew)

def getAnnotationInfo(src):

    masks = sorted(glob(src + "masks/**/*"))
    annotationInfo = []

    idDict = json.load(open(src + "imgDict.json"))
    classDict = json.load(open(src + "classDict.json"))

    for n, m in enumerate(masks):
        imgName = m.split("/")[-1]
        imgClass = m.split("/")[-2]

        # if the class that is being loaded is not in the dictionary 
        # then it is being ignored during this data generation 
        if classDict.get(imgClass) is None:
            continue
        
        printProgressBar(n, len(masks) - 1, prefix = 'AnnotationInfo:', suffix = 'Complete', length=15)
        mask = cv2.imread(m)
        _, segment, area, bbox, _ = getAnnotations(mask)
        annoDict = {}
        annoDict["segmentation"] = segment
        annoDict["area"] = area
        annoDict["iscrowd"] = 0                       # always individual fish
        annoDict["image_id"]= idDict.get(imgName)                      # there is only one segmentation per image so the id is the same as the image
        annoDict["bbox"] = bbox
        annoDict["category_id"] = classDict[imgClass]                   # always fish category
        annoDict["id"] = n                           # iterate through unique images
        annotationInfo.append(annoDict)

    # save the dictionary as a json file in the src
    json.dump(annotationInfo, open(src + "annotations.json", 'w'))

    return(annotationInfo)   

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Fish4Knowledge/"

    getAnnotationInfo(src)

    pass