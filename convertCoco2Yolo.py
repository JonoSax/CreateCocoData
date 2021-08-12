'''
Take the coco structure and convert it to yolo structure
'''

import json
import multiprocessing
import os
from cocoDataStructure.utilities import createIDDict, dirMaker, printProgressBar
from multiprocessing import Pool, Process
from itertools import repeat
from random import shuffle
from overlaySegment import annotateYoloSegments

def createYoloData(src, dest):

    '''
    Take the coco data and convert into yolo

    YOLO data strcuture:

    data 
        images
            train
                img0.png
                img1.png
                ...
            test
                imgn.png
            val
                imgn.png
        labels
            train
                img0.txt
                img1.txt
                ...
            test
                imgn.txt
            val
                imgn.txt

    imgn.txt:
    [class ID] [xCenter%] [yCenter%] [w%] [h%]
    
    '''

    dataType = src.split("/")[-1].split(".")[0]

    cocoData = json.load(open(src))

    labelDest = dest + "labels/" + dataType + "/"
    imgDest = dest + "images/" + dataType + "/"

    dirMaker(labelDest, True)
    dirMaker(imgDest, True)

    images = cocoData["images"]
    # shuffle(images)
    images = images[:100]
    annos = cocoData["annotations"]

    # create the dictionary which relates the image ids to their anno ids
    annoIds = createIDDict(annos, "image_id", "*")

    job = []
    for n, i in enumerate(images):
        job.append(Process(target=convertData, args=(i, annos, annoIds, labelDest, imgDest)))
        job[n].start()
    for n, j in enumerate(job):
        j.join()
        printProgressBar(n, len(images) - 1, "Converted: ", length = 20)

def convertData(i, annos, annoIds, labelDest, imgDest):

    imgPath = i["file_name"]
    imgName = imgPath.split("/")[-1].split(".")[0]
    imgId = i["id"]
    img_h = i["height"]
    img_w = i["width"]

    f = open(os.path.join(labelDest, imgName + ".txt"), "w")

    # copy the image to the new destination
    os.system("cp " + imgPath + " " + imgDest)

    imgAnno = annoIds[imgId]

    for a in imgAnno:
        x, y, h, w = a["bbox"]
        label = a["category_id"]

        # get the centre of the annotation
        x += h/2
        y += w/2

        # normalise the coordinates
        x /= img_w
        y /= img_h
        h /= img_w
        w /= img_h

        f.write(f"{label} {x} {y} {h} {w}\n")

    f.close()


if __name__ == "__main__":

    multiprocessing.set_start_method("fork")

    srcs = ["/Volumes/USB/data/CocoData/train.json",
    "/Volumes/USB/data/CocoData/val.json"]

    dest = "/Volumes/USB/data/YoloData/"

    for src in srcs:
        createYoloData(src, dest)

    yolosrc = "/Volumes/USB/data/YoloData/"

    annotateYoloSegments(yolosrc, False)