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
from numpy.core.fromnumeric import repeat
if __name__ == "__main__":
    from utilities import *
else:
    from cocoDataStructure.utilities import *
import json
import os
import pandas as pd
import cv2
import cv2.aruco as aruco
from multiprocessing import Pool
from itertools import repeat

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

def getSegmentation(img, dsmp = 40):

    '''
    Get the list of points which make up the segmentation mask

    img:        numpy array of the image
    dsmp:       the maximum number of boundary points to use to describe the outline
    '''

    # count all the non-zero pixel positions as the mask
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # raw position info
    pixels = np.where(imgGray != 0)

    # get the top of the image (left to right)
    x0 = []
    y0 = []
    for i in range(imgGray.shape[0]):
        pos = np.where((imgGray[i, :])==255)[0]
        if len(pos) > 0:
            y0.append(max(pos))
            x0.append(i)


    # get the bottom of the image (right to left)
    x1 = []
    y1 = []
    for i in range(imgGray.shape[0]-1, -1, -1):
        pos = np.where((imgGray[i, :])==255)[0]
        if len(pos) > 0:
            y1.append(min(pos))
            x1.append(i)
    
    import matplotlib.pyplot as plt
    if len(x0) + len(x1) > dsmp:
        x0 = np.interp(np.linspace(0, len(x0), int(dsmp/2)), np.arange(len(x0)), np.array(x0))
        y0 = np.interp(np.linspace(0, len(y0), int(dsmp/2)), np.arange(len(y0)), np.array(y0))    
        x1 = np.interp(np.linspace(0, len(x1), int(dsmp/2)), np.arange(len(x1)), np.array(x1))
        y1 = np.interp(np.linspace(0, len(y1), int(dsmp/2)), np.arange(len(y1)), np.array(y1))    

    border = list(np.hstack(np.c_[list(x0) + list(x1), list(y0) + list(y1)]))
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

def createAnnotationDict(idDict, classDict, seg, area, crowd, imgName, bbox, imgClass, annoid):

    '''
    Take in the coco annotation information
    '''

    imId = idDict.get(imgName)
    if imId is None:
        print(f"Invalid Image ID: {imgName}")
        return(None)

    classId = classDict.get(imgClass)
    if classId is None:
        print(f"Invalid Class: {imgClass}")
        return(None)

    annoDict = {}
    annoDict["segmentation"] = seg
    annoDict["area"] = int(area)
    annoDict["iscrowd"] = int(crowd)                       # always individual fish
    annoDict["image_id"] = int(imId)                      # there is only one segmentation per image so the id is the same as the image
    annoDict["bbox"] = bbox
    annoDict["category_id"] = int(classId)                   # always fish category
    annoDict["id"] = int(annoid)                           # iterate through unique images

    return(annoDict)

def getMaskInfo(src, detailSegData = False, segData = True, cpuNo = 12): 

    '''
    If there are masks then process that info

    detailSegData:      If true, the include every pixel as an annotation
    segData             If false, don't include any segmentation data
    '''

    annotationInfo = []

    masks = sorted(glob(src + "masks/**/*"))

    idDict = json.load(open(src + "imgDict.json"))
    classDict = json.load(open(src + "classDict.json"))

    if cpuNo:
        with Pool(cpuNo) as p:
            annotationInfo = p.starmap(maskInfo, zip(masks, repeat(classDict), repeat(detailSegData), repeat(segData), repeat(idDict),))

        # assign a unique annotation id
        for n, _ in enumerate(annotationInfo):
            annotationInfo[n]["id"] = n

    else:   
        for n, m in enumerate(masks):
            annoDict = maskInfo(m, classDict, detailSegData, segData, idDict)
            annotationInfo.append(annoDict)

    return(annotationInfo)

def maskInfo(m, classDict, detailSegData, segData, idDict):

    imgName = m.split("/")[-1]
    imgClass = m.split("/")[-2]

    # if the class that is being loaded is not in the dictionary 
    # then it is being ignored during this data generation 
    if classDict.get(imgClass) is None:
        return
   
    mask = cv2.imread(m)
    _, segment, area, bbox, _ = getAnnotations(mask)

    if not segData:
        segment = ""

    elif detailSegData:
        segment = str(list(np.hstack(np.c_[np.where(mask[:, :, 0])])))[1:-1]
        
    annoPlaceHolder = -1
    annoDict = createAnnotationDict(idDict, classDict, segment, area, 0, imgName, bbox, imgClass, annoPlaceHolder)
    
    return(annoDict)

def processQUTData(src, segData = False):

    '''
    Function to process the QUT data source.

    The QUT data source is super simple. The images are cropped 
    already so the bounding box is the size of the image
    '''

    annotationInfo = []

    images = sorted(glob(src + "images/**/*"))

    idDict = json.load(open(src + "imgDict.json"))
    classDict = json.load(open(src + "classDict.json"))

    # NOTE this can be parallelised
    for n, i in enumerate(images):
        imgName = i.split("/")[-1]
        imgClass = i.split("/")[-2]

        # if the class that is being loaded is not in the dictionary 
        # then it is being ignored during this data generation 
        if classDict.get(imgClass) is None:
            continue
        
        if (n/len(images)*100) % 10 == 0:
            print(f"getAnnotationInfo:{src.split('/')[-2]} - {n}/{len(images)}")
        
        # printProgressBar(n, len(images) - 1, prefix = 'AnnotationInfo:', suffix = '', length=15)
        
        img = cv2.imread(i)
        x, y, _ = img.shape
        
        annoDict = createAnnotationDict(idDict, classDict, "", int(x*y), 0, imgName, f"0,0,{y},{x}", imgClass, n)
        
        if annoDict is not None:
            annotationInfo.append(annoDict)

    return(annotationInfo)

def processOpenImagesData(src):

    '''
    Function to process the openimages data source
    '''

    def getInfo(txtsrc, annotationInfo, max = np.inf):
        for n, i in enumerate(txtsrc):
            if n == 0:
                continue
            if n > max:
                break
            # get the info out of the csv
            ImageID, _, LabelName, _, XMin, XMax, YMin, YMax, _, _, IsGroupOf, _, _, _, _ = i.split(",") 

            label = LabelNameDict[LabelName].lower()

            img = cv2.imread(f"{src}/images/{label}/{ImageID}.jpg")
            y, x, _ = img.shape

            x0 = int(np.double(XMin)*x)
            x1 = int(np.double(XMax)*x)
            y0 = int(np.double(YMin)*y)
            y1 = int(np.double(YMax)*y)

            bbox = f"{x0},{y0},{x1-x0},{y1-y0}"
            area = (x1-x0)*(y1-y0)

            annoDict = createAnnotationDict(idDict, classDict, "", area, IsGroupOf, f"{ImageID}.jpg", bbox, label, n)
        
            if annoDict is not None:
                annotationInfo.append(annoDict)

        return(annotationInfo)

    trainInfo = open(src + "sub-train-annotations-bbox.csv")
    testInfo = open(src + "sub-test-annotations-bbox.csv")
    valInfo = open(src + "sub-validation-annotations-bbox.csv")

    imgs = glob(src + "images/*/*")

    annotationInfo = []

    classDict = json.load(open(src + "classDict.json"))
    idDict = json.load(open(src + "imgDict.json"))
    openImagesDict = open(src + "class-descriptions-boxable.csv")
    LabelNameDict = {}
    for o in openImagesDict:
        k, v = o.split(",")
        LabelNameDict[k] = v.replace("\n", "")


    # read in line by line the csv info because a pandas format is too large
    # in memory
    annotationInfo = getInfo(trainInfo, annotationInfo)
    annotationInfo = getInfo(testInfo, annotationInfo)
    annotationInfo = getInfo(valInfo, annotationInfo)

    return(annotationInfo)

def processCocoData(src):

    '''
    Function to process coco data sources (BrackishWaterImages and Aquarium)
    '''

    return

def processArucoMarkers(src, segData = False):

    '''
    Segment the aruco markers and create their masks
    '''

    images = sorted(glob(src + "images/**/*"))

    arucoDict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)

    idDict = json.load(open(src + "imgDict.json"))

    classDict = json.load(open(src + "classDict.json"))

    annotationInfo = []                

    annoid = 0

    for n, i in enumerate(images):
        if (n/len(images)*100) % 10 == 0:
            print(f"getAnnotationInfo:{src.split('/')[-2]} - {n}/{len(images)}")
        
        # printProgressBar(n, len(images)-1, "getAnnotationInfo", length = 20)

        img = cv2.imread(i)

        imgName = i.split("/")[-1]        
        imgClass = i.split("/")[-2] 

        # create arcuo marker parameters
        param = aruco.DetectorParameters_create()

        # on my mac, takes 5.4 sec to run through the 216 images in blue_aruco_dataSmall
        param.minDistanceToBorder = 0               # allows for markers to be identified at the edge of the frame      -0.1sec overhead (makes it faster)  
        param.adaptiveThreshWinSizeStep = 4         # good compromise bewteen true positive and speed,                  1.7sec overhead
        param.adaptiveThreshWinSizeMax = 70         # good compromise between true positive and speed,                  1.8sec overhead
        param.adaptiveThreshConstant = 4            # good compromise between true positive and speed,                  1.0sec overhead
        param.polygonalApproxAccuracyRate = 0.09    # most accurate parameter, speed is very stable across variables,   0.9sec overhead
        param.perspectiveRemovePixelPerCell = 8     # fastest speed without compromising accuracy,                      -0.1sec overhead
    
        # detect markers
        corners, ids, _ = aruco.detectMarkers(img, arucoDict, parameters = param)

        if corners is not None:

            for c in corners:
                c = c[0].astype(int)
                area = PolyArea(c[:, 0], c[:, 1])        # trapizoid calculation?
                
                yMi, xMi = np.min(c, axis = 0)
                yMa, xMa = np.max(c, axis = 0)

                bbox = f"{yMi},{xMi},{yMa-yMi},{xMa-xMi}"            # is this the same as seg?

                if segData:
                    seg = str(list(np.hstack(c))).replace("[", "").replace("]", "")          # get the boundaries of the box (just the corners???)
                else:
                    seg = ""

                annoDict = createAnnotationDict(idDict, classDict, seg, area, 0, imgName, bbox, imgClass, annoid)
                
                if annoDict is not None:
                    annotationInfo.append(annoDict)
                    annoid += 1
                    
            # imgmod = aruco.drawDetectedMarkers(img.copy(), corners, ids, [0, 0, 255])
            # cv2.imshow(f"imgmod", imgmod); cv2.waitKey(0)

    return(annotationInfo)

def getAnnotationInfo(src):

    # if there are masks then process that info
    if os.path.isdir(src + "masks/"):
        annotationInfo = getMaskInfo(src)
    else:
        if "qut" in src.lower():
            annotationInfo = processQUTData(src)
        elif "openimages" in src.lower():
            annotationInfo = processOpenImagesData(src)
        elif "aruco" in src.lower():
            annotationInfo = processArucoMarkers(src)


    # save the dictionary as a json file in the src
    json.dump(annotationInfo, open(src + "annotations.json", 'w'))

    return(annotationInfo)   

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/Aruco+Net/net_day_shade_pool/"
    src = "/Volumes/WorkStorage/BoxFish/dataStore/netData/foregrounds/mod/"
    src = "/Volumes/USB/data/YOLO_data/YOLO_data/Ulucan/"
    src = "/media/boxfish/USB/data/CocoData/Ulucan/"
    getAnnotationInfo(src)  