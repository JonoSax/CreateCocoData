import numpy as np
from glob import glob 
import json
import cv2
from multiprocessing import Process
import multiprocessing
import os

def drawLine(img, point0, point1, blur = 2, colour = [0, 0, 255]):

    # draw a line between two points
    # Inputs:   (img), image to draw on
    #           (point#), the points to draw between, doesn't matter which 
    #               order they are specified
    #           (blur), the thickness of the line
    #           (colour), colour of the line
    # Outputs:  (img), image with the line drawn

    # get the distance between the points
    dist = np.ceil(np.sqrt(np.sum(abs(point1 - point0)**2)))

    x, y, _ = img.shape

    # interpolate for the correct number of pixels between points
    xp = np.clip(np.linspace(int(point0[1]), int(point1[1]), int(dist)).astype(int), 0, x-blur)
    yp = np.clip(np.linspace(int(point0[0]), int(point1[0]), int(dist)).astype(int), 0, y-blur)


    # change the colour of these pixels which indicate the line
    for vx in range(-blur, blur, 1):
        for vy in range(-blur, blur, 1):
            img[xp+vx, yp+vy, :] = colour

    # NOTE this may do the interpolation between the points properly!
    # pos = np.linspace(point0.astype(int), point1.astype(int), int(dist)).astype(int)
    
    
    return(img)

def createIDDict(targetdict, keytype, classtype, pathsrc = ""):

    '''
    Create a dictionary which extracts information from within a list of dictionaries
    '''

    imgDict = {}

    for i in targetdict:
        key = i[keytype]
        if classtype == "*":
            if imgDict.get(key) is None:
                imgDict[key] = []
            imgDict[key].append(i)
        else:
            imgDict[key] = pathsrc + str(i[classtype])

    return(imgDict)

def getSegPos(seg):

    '''
    Convert the coco pixel positions into numpy ones
    '''

    pixX = []
    pixY = []
    for n, s in enumerate(seg):

        # for every 2nd value, create a new numpy entry
        if n%2 == 0:
            pixX.append(int(np.round(s)))
        else:
            pixY.append(int(np.round(s)))

    pix = np.c_[pixX, pixY]

    return(pix)

if __name__ == "__main__":

    imgsrc = "data/coco/images/"
    segsrc = "data/coco/annotations/instances_train2017.json"


    imgsrc = "data/datasources/testdataset/fish/"
    segsrc = "data/datasources/testdataset/cocoInfo_Val.json"

    imgsrc = "/Volumes/USB/segmentationData/instance/Fish4Knowledge/testdataset2/fish/"
    segsrc = imgsrc + "train.json"

    imgsrc = "/Volumes/USB/segmentationData/dataGenerator/foregrounds/modImages/img/"
    segsrc = imgsrc + "train.json"


    segFile = open(segsrc)
    segments = json.load(segFile)

    annotations = segments["annotations"]
    images = segments["images"]
    categories = segments["categories"]

    # downloadImages(imgsrc, images, limit = 10)
    imgDict = createIDDict(images, "id", "file_name", imgsrc)
    # imgDict = createIDDict(images, "id", "file_name")
    categories = createIDDict(categories, "id", "name")


    for n, a in enumerate(annotations):
        id = a["image_id"]
        catid = a["category_id"]
        imgpath = imgDict.get(id)
        if imgpath is None:
            continue

        cat = categories[catid]
        img = cv2.imread(imgpath)

        if img is None:
            continue

        print(f"ID = {cat}")
        imgm = img.copy()
        imgm = (imgm.astype(float)*0.2).astype(np.uint8)
        for s in a["segmentation"]:
            x0, y0, x1, y1 = a["bbox"]
            x1+=x0
            y1+=y0
            pix = getSegPos(s)
            '''
            for p in pix:
                imgm[p[0], p[1]] = [0, 0, 255]
            '''
            
            for p0, p1 in zip(pix, np.vstack([pix[1:], pix[0]])):
                try:
                    imgm = drawLine(imgm, p0, p1)
                except:
                    pass
            
            imgm = drawLine(imgm, np.array([x0, y0]), np.array([x1, y0]))
            imgm = drawLine(imgm, np.array([x0, y0]), np.array([x0, y1]))
            imgm = drawLine(imgm, np.array([x1, y1]), np.array([x1, y0]))
            imgm = drawLine(imgm, np.array([x1, y1]), np.array([x0, y1]))
        cv2.imshow("img", np.hstack([img, imgm])); cv2.waitKey(0)

        # y0, x0, y1, x1 = np.array(bbox.replace("[", "").replace("]", "").replace(" ", "").split(",")).astype(int)
        # cv2.rectangle(imgm, [x0, y0], [x0+x1, y0+y1], [0, 0, 255], 2)
