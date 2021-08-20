'''
From the bounding box info in the coco info, create segmentations if they 
do not already exist
'''

import numpy as np
import cv2
from glob import glob
from cocoDataStructure.utilities import getAllImages, createIDDict
import json
from cocoDataStructure.annotations import getSegmentation, processCoco
from multiprocessing.pool import Pool
from itertools import repeat

def getbbox(bbox):

    '''
    Get the bounding box in the format for grabcut
    '''
    
    bbox = [int(b) for b in bbox.split(",")]

    return(bbox)

def segmentImg(i, annoInfo, iter = 10):

    '''
    From a bounding box, get the segmentation of an image
    '''

    imgpath = i["file_name"]
    img = cv2.imread(imgpath)
    if img is None:
        return

    imgId = i["id"]
    imgAnno = annoInfo.get(imgId)
    if imgAnno is None:
        return

    x, y, _ = img.shape
    mask = np.zeros((x,y), dtype="uint8")
    fgModel = np.zeros((1, 65), dtype="float")
    bgModel = np.zeros((1, 65), dtype="float")


    for a in imgAnno:
        annoId = a["id"]
        rect = getbbox(a['bbox'])

        (mask, bgModel, fgModel) = cv2.grabCut(img, mask, rect, bgModel,
        fgModel, iterCount=iter, mode=cv2.GC_INIT_WITH_RECT)

        '''
        # loop over the possible GrabCut mask values
        for (name, value) in values:
            # construct a mask that for the current value
            print("[INFO] showing mask for '{}'".format(name))
            valueMask = (mask == value).astype("uint8") * 255
            # display the mask so we can visualize it
            cv2.imshow(name, valueMask)
            cv2.waitKey(0)
        '''

        outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
        outputMask = (outputMask * 255).astype("uint8")
        _, border = getSegmentation(outputMask, 40)
        segment = str(list(np.hstack(border)))
        segment = processCoco(segment)
        return(annoId, segment)


if __name__ == "__main__":

    cpuNo = 1

    src = "/media/boxfish/USB/data/CocoData/NorFisk_v1.0/"

    annoInfo = json.load(open(src + "annotations.json"))
    imgInfo = json.load(open(src + "images.json"))

    annoInfoId = createIDDict(annoInfo, "image_id", "*")

    values = (
	("Definite Background", cv2.GC_BGD),
	("Probable Background", cv2.GC_PR_BGD),
	("Definite Foreground", cv2.GC_FGD),
	("Probable Foreground", cv2.GC_PR_FGD))

    if cpuNo > 1:
        with Pool(14) as p:
            segmentation = p.starmap(segmentImg, zip(imgInfo, repeat(annoInfoId)))
    else:
        segmentation = []
        for i in imgInfo[:10]:
            segmentation.append(segmentImg(i, annoInfoId))

    segDict = {}
    for id, seg in segmentation:
        segDict[id] = seg

    for a in annoInfo:
        a["segmentation"] = segDict.get(a["id"], "")

    json.dump(annoInfo, open(src + "annotationsSegment.json", 'w'))
    print("done")
