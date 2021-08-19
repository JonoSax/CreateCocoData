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
import numpy as np

def createYoloData(src, dest, parallel = True):

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
    shuffle(images)
    annos = cocoData["annotations"]

    # create the dictionary which relates the image ids to their anno ids
    annoIds = createIDDict(annos, "image_id", "*")


    # to ensure that not too many processes are open, divide the data into sections and process
    if parallel:
        sectLen = 2000
        for imn in range(int(np.ceil(len(images)/sectLen))):
            imageSect = images[imn*sectLen:(imn+1)*sectLen]
            job = []
            for n, i in enumerate(imageSect):
                job.append(Process(target=convertData, args=(i, annos, annoIds, labelDest, imgDest)))
                job[n].start()
            for n, j in enumerate(job):
                j.join()
            print(imn * sectLen)

    else:
        for i in images:
            convertData(i, annos, annoIds, labelDest, imgDest)
    print("Done")
    
def convertData(i, annos, annoIds, labelDest, imgDest):

    imgPath = i["file_name"].replace("Volumes", "media/boxfish")    # NOTE hardcode replace the path
    
    # get the image name and remove the prefix
    imgName = imgPath.split("/")[-1]
    imgPrefix = f'.{imgName.split(".")[-1]}'
    imgName = imgName.replace(imgPrefix, "")

    imgId = i["id"]
    img_h = i["height"]
    img_w = i["width"]

    imgAnno = annoIds.get(imgId, False)
    if imgAnno:
        f = open(os.path.join(labelDest, imgName + ".txt"), "w")

        # copy the image to the new destination
        os.system("cp " + imgPath + " " + imgDest)


        for a in imgAnno:
            x, y, h, w = a["bbox"]
            label = a["category_id"] - 1

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

    wd = "/Volumes/USB/data/"
    wd = "/media/boxfish/USB/data/"

    srcs = [wd + "CocoData/train.json",
    wd + "CocoData/val.json",
    wd + "CocoData/test.json"]

    yolosrc = wd + "YoloDataBrackish/"
    yolosrc = wd + "YoloData/"
    yolosrc = wd + "YoloDataGloryBay/"


    for src in srcs:
        createYoloData(src, yolosrc, True)


    annotateYoloSegments(yolosrc, True)