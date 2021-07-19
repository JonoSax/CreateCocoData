from glob import glob
import json
import numpy as np
from random import random
from itertools import repeat

from cocoDataStructure.utilities import printProgressBar
from cocoDataStructure.categories import getCategoriesInfo


def getDataSplit(src, split = [0.8, 0.1, 0.1]):

    '''
    From the images and segmentation masks, create a training data and 
    validation data split
    '''

    print("Running getDataSplit")

    # get the split 
    train, val, test = split

    if train+val+test < 1:
        test = 1 - (train + val)

    # if the sum is greater than 1 then normalize
    if train+val+test > 1:
        splitSum = train+val+test
        train /= splitSum
        val /= splitSum
        test /= splitSum

    cocoAll = json.load(open(src + "cocoAll.json", "r"))

    trainCoco = cocoAll.copy(); trainCoco["images"] = []
    valCoco = cocoAll.copy(); valCoco["images"] = []
    testCoco = cocoAll.copy(); testCoco["images"] = []

    imgs = cocoAll["images"]

    for i in imgs:
        r = random() 
        if r < train:
            trainCoco['images'].append(i)
        elif r < train + val:
            valCoco['images'].append(i)
        else:
            testCoco['images'].append(i)

    json.dump(trainCoco, open(src + "trainCoco.json", "w"))
    json.dump(valCoco, open(src + "valCoco.json", "w"))
    json.dump(testCoco, open(src + "testCoco.json", "w"))

    print("Finished getDataSplit")


def createCocoData(src):

    '''    
    create the master dictionary to put all info
    '''

    print("Running createCocoData")

    cocoInfo = {
        "info":{},
        "licenses":{},
        "images":{},
        "annotations":{},
        "categories":{},
    }

    # create info and assign to master dict
    cocoInfo["info"] = {
        "description": "Fish dataset",
        "url": ["https://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/", 
        "https://www.kaggle.com/sripaadsrinivasan/fish-species-image-data", 
        "https://www.kaggle.com/crowww/a-large-scale-fish-dataset"],
        "version": "1.0",
        "year": 2021,
        "contributor": "Jonathan Reshef",
        "date_created": "2021/06/10"
    }

    cocoInfo["categories"] = json.load(open(src + "categories.json"))

    imgInfoAll = []
    imgJson = sorted(glob(src + "*/images.json"))
    lastID = 0          # ensure the ids are uniquely assigned
    for i in imgJson:
        imgInfo = json.load(open(i))
        [exec(f'i["id"] += {lastID}') for i in imgInfo]       # adjust the ID's
        lastID = imgInfo[-1]["id"]
        imgInfoAll += imgInfo

    cocoInfo["images"] = imgInfoAll

    annotationInfoAll = []
    annosJson = sorted(glob(src + "*/annotations.json"))
    lastImgID = 0          # ensure the image ids are uniquely assigned
    lastObjID = 0       # ensure the annotation id's are uniquely assigned
    for a in annosJson:
        annosInfo = json.load(open(a))
        [exec(f'i["image_id"] += {lastImgID}') for i in annosInfo] 
        [exec(f'i["id"] += {lastObjID}') for i in annosInfo] 
        [exec(f'i["bbox"] = [int(ib) for ib in i["bbox"].split(",")]') for i in annosInfo] 
        [exec(f'i["segmentation"] = []') for i in annosInfo] 
        lastImgID = annosInfo[-1]["image_id"]
        lastObjID = annosInfo[-1]["id"]
        annotationInfoAll += annosInfo

    cocoInfo["annotations"] = annotationInfoAll

    json.dump(cocoInfo, open(src + "cocoAll.json", "w"), indent=4)

    print("Finished createCocoData")


def combineCocoData(dataPath, dest):

    '''
    Take a list of coco annotations and combine them

    dataPath:   list of paths to the coco files to combine
    dest:       path to where the combined data will be stored

    '''
    keys = ['info', 'licenses', 'images', 'annotations', 'categories']
    info = dict.fromkeys(keys, [])

    for d in dataPath:
        cocoInfo = json.load(open(d))
        info.update(cocoInfo)

    return

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/"

    # categoryInfo = getCategoriesInfo(src)
    createCocoData(src)

    getDataSplit(src)
