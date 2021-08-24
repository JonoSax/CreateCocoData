'''
From the bounding box info in the coco info, create segmentations if they 
do not already exist
'''

import numpy as np
import cv2
from glob import glob
from cocoDataStructure.utilities import createIDDict, PolyArea
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

    segment = []
    area = []
    annoIds = []
    info = []
    for a in imgAnno:
        if a["iscrowd"] == 1:
            continue
        annoIds = a["id"]
        rect = getbbox(a['bbox'])

        (mask, bgModel, fgModel) = cv2.grabCut(img, mask, rect, bgModel,
        fgModel, iterCount=iter, mode=cv2.GC_INIT_WITH_RECT)

        
        try:
            outputMask = np.where((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD), 0, 1)
            outputMask = (outputMask * 255).astype("uint8")
            _, border = getSegmentation(outputMask, 40)

            for b in border:
                cv2.circle(img, (int(b[0]), int(b[1])), 4, [255, 0, 0], 4)
            cv2.rectangle(img, (rect[0], rect[1]), (rect[2]+rect[0], rect[3]+rect[1]), [0, 0, 255], 4)
            cv2.imshow("img", img); cv2.waitKey(0)

            segmentRaw = str(list(np.hstack(border)))
            segment = processCoco(segmentRaw)
            area = PolyArea(border[:, 0], border[:, 1])
            info.append([annoIds, segment, area])
        except:
            pass

    return(info)



if __name__ == "__main__":

    cpuNo = 1

    src = "/media/boxfish/USB/data/CocoData/NorFisk_v1.0/"
    src = "/media/boxfish/USB/data/CocoData/openimages/"

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
            segmentation = p.starmap(segmentImg, zip(imgInfo[2:1000], repeat(annoInfoId)))
    else:
        segmentation = []
        for i in imgInfo[10:]:
            segmentation.append(segmentImg(i, annoInfoId))

    segDict = {}
    for info in segmentation:
        for i in info:
            if i is not None: 
                annoid, seg, area = i
                segDict[annoid] = [seg, area]

    for a in annoInfo:
        # add the segmentation annotations and update the area
        info = segDict.get(a["id"])
        if info is not None:
            a["segmentation"], a["area"] = info

    json.dump(annoInfo, open(src + "annotationsSegment.json", 'w'))
    print("done")
