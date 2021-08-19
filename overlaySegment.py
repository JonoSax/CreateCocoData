'''
Visualise the coco data sets
'''

import numpy as np
from glob import glob 
import json
import cv2
from multiprocessing import Process
import multiprocessing
import os
from cocoDataStructure.utilities import createIDDict
from random import shuffle
from cocoDataStructure.utilities import *

def getSegPos(seg):

    '''
    Convert the coco pixel positions into numpy ones
    '''

    pixX = []
    pixY = []
    try:
        for n, s in enumerate(seg):

            # for every 2nd value, create a new numpy entry
            if n%2 == 0:
                pixX.append(int(np.round(s)))
            else:
                pixY.append(int(np.round(s)))

        pix = np.c_[pixX, pixY]

    except:
        pix = []

    return(pix)

def annotateCocoSegments(src, imgsrc = "", random = True):

    '''
    Load in the coco information and annotate the segments

    Inputs   
    src:        the coco file containing the image and segment info
    random      if set to a value then will randomally n number of annotations

    '''
    # src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Aquarium/test/"
    # jsonsrc = src+"_annotations.coco.json"
    segments = json.load(open(src, "r"))
    annotations = segments["annotations"]
    images = segments["images"]
    # [exec(f'i["file_name"] = "{src}{i["file_name"]}"') for i in images]
    categories = segments["categories"]
    
    if random:
        shuffle(images)

    annosImageId = createIDDict(annotations, "image_id", "*")

    # downloadImages(imgsrc, images, limit = 10)
    imgDict = createIDDict(images, "id", "file_name")
    # imgDict = createIDDict(images, "id", "file_name")
    categories = createIDDict(categories, "id", "name")

    keys = list(imgDict.keys())[-10:]
    [imgDict[k] for k in keys]
    randomCount = 0

    for n, a in enumerate(images):
        
        id = a["id"]
        annos = annosImageId.get(id)
        if annos is None:
            continue

        imgpath = imgDict.get(id)
        if imgpath is None:
            continue

        img = cv2.imread(f"{imgsrc}{imgpath}")
        imgm = img.copy()

        for a in annos:
            catid = a["category_id"]
            cat = categories[catid]

            print(f"ID = {cat}")
            # imgm = (imgm.astype(float)*0.2).astype(np.uint8)
            for s in a["segmentation"]:
                pix = getSegPos(s)
                
                for p in pix:
                    try:
                        cv2.circle(imgm, p.astype(int), 5, [255, 0, 0], 4)
                    except:
                        pass
                '''
                # draw the outline of the annotation
                for p0, p1 in zip(pix, np.vstack([pix[1:], pix[0]])):
                    try:
                        imgm = drawLine(imgm, p0, p1)
                    except:
                        pass
                '''

            # draw the bounding box
            x0, y0, x1, y1 = a["bbox"]
            x1+=x0
            y1+=y0
            imgm = drawLine(imgm, np.array([x0, y0]), np.array([x1, y0]))
            imgm = drawLine(imgm, np.array([x0, y0]), np.array([x0, y1]))
            imgm = drawLine(imgm, np.array([x1, y1]), np.array([x1, y0]))
            imgm = drawLine(imgm, np.array([x1, y1]), np.array([x0, y1]))
            
            xp = int(np.median([x0, x1]))
            yp = int(np.median([y0, y1]))
            
            cv2.putText(imgm, str(f"Crowd: {a['iscrowd']}"), (xp, yp), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
            cv2.putText(imgm, str(f"Class: {cat}"), (xp, yp+50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)
        imgC = np.hstack([img, imgm])
        xc, yc, _ = imgC.shape
        if xc*yc > 3e6:
            r = 5e6/(xc*yc)
            imgC = cv2.resize(imgC, (int(imgC.shape[1]*r), int(imgC.shape[0]*r)))

        cv2.imshow("img", imgC); cv2.waitKey(0)
        randomCount += 1

        # y0, x0, y1, x1 = np.array(bbox.replace("[", "").replace("]", "").replace(" ", "").split(",")).astype(int)
        # cv2.rectangle(imgm, [x0, y0], [x0+x1, y0+y1], [0, 0, 255], 2)

def annotateYoloSegments(src, random = True):

    '''
    Load in the YOLO information and annotate the segments

    Inputs   
    src:        the yolo file containing the image and segment info
    random      if set to a value then will randomally n number of annotations

    '''
    # src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Aquarium/test/"
    # jsonsrc = src+"_annotations.coco.json"

    images = sorted(glob(src+"images/*/*.*"))

    # annotations must be he same name as the images (except the suffix)
    annos = [i.replace(f".{i.split('/')[-1].split('.')[-1]}", ".txt").replace("images", "labels") for i in images]

    if random:
        ids = np.arange(len(images))
        np.random.shuffle(ids)
        annos = [annos[i] for i in ids]
        images = [images[i] for i in ids]

    for n, (i, a) in enumerate(zip(images, annos)):

        img = cv2.imread(i)
        y, x, _ = img.shape
        imgm = img.copy()

        annos = open(a, "r").readlines()

        for anno in annos:

            anno = anno.replace("\n", "")
            cat, xC, yC, w, h = anno.split(" ")

            print(f"class = {cat}")

            # get the corners of the bounding box
            x0 = int((float(xC) - float(w)/2) * x)
            y0 = int((float(yC) - float(h)/2) * y)
            x1 = int((float(xC) + float(w)/2) * x)
            y1 = int((float(yC) + float(h)/2) * y)

            # draw the bounding box
            imgm = drawLine(imgm, np.array([x0, y0]), np.array([x1, y0]))
            imgm = drawLine(imgm, np.array([x0, y0]), np.array([x0, y1]))
            imgm = drawLine(imgm, np.array([x1, y1]), np.array([x1, y0]))
            imgm = drawLine(imgm, np.array([x1, y1]), np.array([x0, y1]))
            
            xp = int(np.median([x0, x1]))
            yp = int(np.median([y0, y1]))
            cv2.putText(imgm, str(f"Class: {cat}"), (xp, yp), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 2)

            
        imgC = np.hstack([img, imgm])
        xc, yc, _ = imgC.shape
        if xc*yc > 5e6:
            r = 5e6/(xc*yc)
            imgC = cv2.resize(imgC, (int(imgC.shape[1]*r), int(imgC.shape[0]*r)))

        cv2.imshow("img", imgC); cv2.waitKey(0)

        # y0, x0, y1, x1 = np.array(bbox.replace("[", "").replace("]", "").replace(" ", "").split(",")).astype(int)
        # cv2.rectangle(imgm, [x0, y0], [x0+x1, y0+y1], [0, 0, 255], 2)


if __name__ == "__main__":

    cocosrc = "/Volumes/WorkStorage/BoxFish/dataStore/Aruco+Net/cocoAll.json"
    cocosrc = "/Volumes/WorkStorage/BoxFish/dataStore/netData/foregrounds/cocoAll.json"
    
    cocosrc = "/home/boxfish/Documents/models/yolactMod/data/datasources/coco/train17.json"
    imgsrc = "/home/boxfish/Documents/models/yolactMod/data/datasources/coco/train17_images/"



    cocosrc = "/media/boxfish/USB/Kingston/testVids/extract/BigGloryBayExtractAll.json"
    imgsrc = "/media/boxfish/USB/Kingston/testVids/extract/"

    cocosrc = "/media/boxfish/USB/data/CocoData/cocoAll.json"
    imgsrc = ""

    # annotateCocoSegments(cocosrc, imgsrc)


    yolosrc = "/Volumes/USB/data/coco128/"
    yolosrc = "/media/boxfish/USB/data/YoloDataGloryBay/"
    annotateYoloSegments(yolosrc, False)