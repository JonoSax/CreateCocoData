'''
From all data sources, create the coco data set
'''

from glob import glob
import json
import numpy as np
from random import random
from itertools import repeat

from cocoDataStructure.utilities import printProgressBar
from cocoDataStructure.categories import getCategoriesInfo
from cocoDataStructure.utilities import createIDDict

def createCocoData(src, segData = False):

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
        "https://www.kaggle.com/crowww/a-large-scale-fish-dataset", 
        "https://storage.googleapis.com/openimages/web/visualizer/index.html?set=train&type=segmentation&r=false&c=%2Fm%2F0cnyhnx"],
        "version": "1.0",
        "year": 2021,
        "contributor": "Jonathan Reshef",
        "date_created": "2021/06/10"
    }

    cocoInfo["categories"] = json.load(open(src + "categories.json"))

    imgInfoAll = []
    annotationInfoAll = []
    imgJson = sorted(glob(src + "*/images.json"))
    annosJson = sorted(glob(src + "*/annotations.json"))
    print(f'Using: {", ".join([i.split("/")[-2] for i in imgJson])}')
    lastImgID = 0          # ensure the ids are uniquely assigned
    lastObjID = 0
    for n, (i, a) in enumerate(zip(imgJson, annosJson)):

        print(f'Processing {i.split("/")[-2]}')

        imgInfo = json.load(open(i))
        annosInfo = json.load(open(a))

        [exec(f'i["id"] += {lastImgID}') for i in imgInfo]       # adjust the ID's
        [exec(f'i["image_id"] += {lastImgID}') for i in annosInfo] 
        [exec(f'i["id"] += {lastObjID}') for i in annosInfo] 
        [exec(f'i["bbox"] = [int(ib) for ib in i["bbox"].split(",")]') for i in annosInfo] 
        
        if segData:
            try:
                # if there are segmentations
                [exec(f'i["segmentation"] = [[float(ib) for ib in i["segmentation"].split(",")]]') for i in annosInfo] 
            except:
                # if there are no segmentations
                [exec(f'i["segmentation"] = []') for i in annosInfo] 
        else:
            [exec(f'i["segmentation"] = []') for i in annosInfo] 

        [exec(f'i["iscrowd"] = int(i["iscrowd"])') for i in annosInfo] 

        lastImgID = np.max([i["id"] for i in imgInfo])
        lastObjID = np.max([a["id"] for a in annosInfo])
        
        imgInfoAll += imgInfo
        annotationInfoAll += annosInfo


    cocoInfo["images"] = imgInfoAll
    cocoInfo["annotations"] = annotationInfoAll


    json.dump(cocoInfo, open(src + "cocoAll.json", "w"), indent=4)

    print("     Finished createCocoData")

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

    trainCoco = cocoAll.copy(); trainCoco["images"] = []; trainCoco["annotations"] = []
    valCoco = cocoAll.copy(); valCoco["images"] = []; valCoco["annotations"] = []
    testCoco = cocoAll.copy(); testCoco["images"] = []; testCoco["annotations"] = []

    imgs = cocoAll["images"]

    annoId = createIDDict(cocoAll["annotations"], 'image_id', "*")

    for i in imgs:
        r = random() 
        imId = i['id']
        if r < train:
            trainCoco['images'].append(i)
            for a in annoId.get(imId, []):
                trainCoco['annotations'].append(a)
        elif r < train + val:
            valCoco['images'].append(i)
            for a in annoId.get(imId, []):
                valCoco['annotations'].append(a)
        else:
            testCoco['images'].append(i)
            for a in annoId.get(imId, []):
                testCoco['annotations'].append(a)

    json.dump(trainCoco, open(src + "train.json", "w"))
    json.dump(valCoco, open(src + "val.json", "w"))
    json.dump(testCoco, open(src + "test.json", "w"))

    print("     Finished getDataSplit")

if __name__ == "__main__":

    src = "/Volumes/WorkStorage/BoxFish/dataStore/Aruco+Net/"
    src = "/Volumes/WorkStorage/BoxFish/dataStore/netData/foregrounds/"
    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/"
    src = "/media/boxfish/USB/data/CocoData/"
    src = "/Volumes/USB/data/CocoData/"

    # categoryInfo = getCategoriesInfo(src)
    createCocoData(src, True)

    getDataSplit(src)
