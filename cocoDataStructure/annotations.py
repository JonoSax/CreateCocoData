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
import os

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

    # get the top of the image (left to right)
    border = []
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

def getMaskInfo(src): 

    '''
    If there are masks then process that info
    '''

    annotationInfo = []

    masks = sorted(glob(src + "masks/**/*"))

    idDict = json.load(open(src + "imgDict.json"))
    classDict = json.load(open(src + "classDict.json"))

    for n, m in enumerate(masks):
        imgName = m.split("/")[-1]
        imgClass = m.split("/")[-2]

        # if the class that is being loaded is not in the dictionary 
        # then it is being ignored during this data generation 
        if classDict.get(imgClass) is None:
            continue
        
        printProgressBar(n, len(masks) - 1, prefix = 'AnnotationInfo:', suffix = '', length=15)
        mask = cv2.imread(m)
        _, segment, area, bbox, _ = getAnnotations(mask)
        annoDict = {}
        annoDict["segmentation"] = ""
        annoDict["area"] = area
        annoDict["iscrowd"] = 0                       # always individual fish
        annoDict["image_id"]= idDict.get(imgName)                      # there is only one segmentation per image so the id is the same as the image
        annoDict["bbox"] = bbox
        annoDict["category_id"] = classDict[imgClass]                   # always fish category
        annoDict["id"] = n                           # iterate through unique images
        annotationInfo.append(annoDict)

    return(annotationInfo)

def processQUTData(src):

    '''
    Function to process the QUT data source.

    The QUT data source is super simple. The images are cropped 
    already so the bounding box is the size of the image
    '''

    annotationInfo = []

    images = sorted(glob(src + "images/**/*"))

    idDict = json.load(open(src + "imgDict.json"))
    classDict = json.load(open(src + "classDict.json"))

    for n, i in enumerate(images):
        imgName = i.split("/")[-1]
        imgClass = i.split("/")[-2]

        # if the class that is being loaded is not in the dictionary 
        # then it is being ignored during this data generation 
        if classDict.get(imgClass) is None:
            continue
        
        printProgressBar(n, len(images) - 1, prefix = 'AnnotationInfo:', suffix = '', length=15)
        img = cv2.imread(i)
        x, y, _ = img.shape
        annoDict = {}
        annoDict["segmentation"] = ""
        annoDict["area"] = int(x*y)
        annoDict["iscrowd"] = 0                       # always individual fish
        annoDict["image_id"]= idDict.get(imgName)                      # there is only one segmentation per image so the id is the same as the image
        annoDict["bbox"] = f"0,0,{y},{x}"
        annoDict["category_id"] = classDict[imgClass]                   # always fish category
        annoDict["id"] = n                  
        annotationInfo.append(annoDict)

    return(annotationInfo)

def processOpenImagesData(src):

    '''
    Function to process the openimages data source
    '''

    return

def processCocoData(src):

    '''
    Function to process coco data sources (BrackishWaterImages and Aquarium)
    '''

    return

def getAnnotationInfo(src):

    # if there are masks then process that info
    if os.path.isdir(src + "masks/"):
        annotationInfo = getMaskInfo(src)
    else:
        annotationInfo = processQUTData(src)

    # save the dictionary as a json file in the src
    json.dump(annotationInfo, open(src + "annotations.json", 'w'))

    return(annotationInfo)   

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/QUT/"

    getAnnotationInfo(src)