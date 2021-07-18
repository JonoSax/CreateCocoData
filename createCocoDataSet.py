from glob import glob
import os
import json
import cv2
import numpy as np
import re
from random import random
from downloadCocoData import printProgressBar


def getDataSplit(imgs, masks, split):

    '''
    From the images and segmentation masks, create a training data and 
    validation data split
    '''

    imgs = sorted(imgs)
    masks = sorted(masks)

    trainImg = []
    trainMask = []
    valImg = []
    valMask = []
    for i, m in zip(imgs, masks):
        if random() < split:
            trainImg.append(i)
            trainMask.append(m)
        else:
            valImg.append(i)
            valMask.append(m)

    return(trainImg, trainMask, valImg, valMask)

def getAnnotationInfo(img):

    '''
    Gets all the mask info for the annotaiton
    '''

    dims = getDims(img)
    pixels, segment = getSegmentation(img)
    bbox = getBoundingBox(pixels)
    area = getArea(pixels)

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


    border = []
    # get the top of the image (left to right)
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
    cocoInfoNew = cocoInfo[0].replace('"[', '[').replace(']"', ']')

    return(cocoInfoNew)

def createCocoData(masks, imgs, dest):

    '''    
    create the master dictionary to put all info
    '''

    cocoInfo = {
        "info":{},
        "licenses":{},
        "images":{},
        "annotations":{},
        "categories":{},
    }

    # create info and assign to master dict
    info = {
        "description": "Fish dataset",
        "url": "https://groups.inf.ed.ac.uk/f4k/GROUNDTRUTH/",
        "version": "1.0",
        "year": 2013,
        "contributor": "Fish4Knowledge, organised by Jonathan Reshef",
        "date_created": "2021/06/10"
    }

    # create the classes and assign to master dict
    categories = [
        # fish
        {
            "id": 1,
            "name": "fish",
            "supercategory": "creatures"
        }
    ]
    '''
    # dock
    {
        "supercategory": "structure",
        "id": 1, 
        "name": "dock"
    }
    '''

    # create per image info 
    images = []
    annotations = []
    for id, (m, i) in enumerate(zip(masks, imgs)):
        
        printProgressBar(id, len(masks) - 1, f"Annos made for {dest.split('/')[-1]}", length = 10)

        # get informaiton from the mask
        img = cv2.imread(i)
        mask = cv2.imread(m)

        # NOTE be careful about the x and y positions of the bbox and pixels.... 
        pixels, segment, area, bbox, dims = getAnnotationInfo(mask)
        imgm = (img.astype(float)*0.1).astype(np.uint8)
        pix = np.c_[pixels]
        for p in pix:
            imgm[p[0], p[1]] *= 10

        # y0, x0, y1, x1 = np.array(bbox.replace("[", "").replace("]", "").replace(" ", "").split(",")).astype(int)
        # cv2.rectangle(imgm, [x0, y0], [x0+x1, y0+y1], [0, 0, 255], 2)

        annotation = {}
        annotation["segmentation"] = segment
        annotation["area"] = area
        annotation["iscrowd"] = 0                       # always individual fish
        annotation["image_id"]= id                      # there is only one segmentation per image so the id is the same as the image
        annotation["bbox"] = bbox
        annotation["category_id"] = 1                   # always fish category
        annotation["id"] = id                           # iterate through unique images

        annotations.append(annotation)

        '''
        id – (Not required) The identifier for the annotation.
        image_id – (Required) Corresponds to the image id in the images array.
        category_id – (Required) The identifier for the label that identifies the object within a bounding box. It maps to the id field of the categories array.
        iscrowd – (Not required) Specifies if the image contains a crowd of objects.
        segmentation – (Not required) Segmentation information for objects on an image. Amazon Rekognition Custom Labels doesn't support segmentation.
        area – (Not required) The area of the annotation.
        bbox – (Required) Contains the coordinates, in pixels, of a bounding box around an object on the image.
        '''
        image = {}
        image["id"] = id                                # only one annotation per image
        image["height"], image["width"] = dims
        image["file_name"] = i.split("/")[-1]           # only the name, no path
        image["date_captured"] = "null"                 # no info on data

        images.append(image)

        '''
        id – (Required) A unique identifier for the image. The id field maps to the id field in the annotations array (where bounding box information is stored).
        license – (Not Required) Maps to the license array.
        coco_url – (Optional) The location of the image.
        flickr_url – (Not required) The location of the image on Flickr.
        width – (Required) The width of the image.
        height – (Required) The height of the image.
        file_name – (Required) The image file name. In this example, file_name and id match, but this is not a requirement for COCO datasets.
        date_captured –(Required) the date and time the image was captured.
        '''


    # assign all created info
    cocoInfo["info"] = info
    cocoInfo["categories"] = categories
    cocoInfo["images"] = images
    cocoInfo["annotations"] = annotations
    cocoInfo["licenses"] = []

    cocoDraftPath = f"{dest}_DRAFT.json"; cocoDraftFile = open(cocoDraftPath, "w")
    json.dump(cocoInfo, cocoDraftFile); cocoDraftFile.close()

    cocoInfoDraft = open(cocoDraftPath)
    cocoFinalPath = f"{dest}.json"; cocoFinalFile = open(cocoFinalPath, "w")
    
    cocoInfoJson = cocoInfoDraft.readlines()
    cocoFinalInfo = processCoco(cocoInfoJson)

    # MAKE THIS SAVE
    cocoFinalFile.write(cocoFinalInfo); cocoFinalFile.close()
    
    print(f"json file for {dest} complete")

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

    src = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/instance/Fish4Knowledge/"


    # load all segmentation masks
    masksrc = "data/datasources/testdataset/mask/"
    masksrc = "/Volumes/USB/segmentationData/instance/Fish4Knowledge/testdataset2/mask/"
    masksrc = "/Volumes/USB/segmentationData/dataGenerator/foregrounds/modImages/mask/"
    masksrc = src + "mask_image_all/"
    masks = glob(masksrc + "*png")
    
    imgsrc = "data/datasources/testdataset/fish/"
    imgsrc = "/Volumes/USB/segmentationData/instance/Fish4Knowledge/testdataset2/fish/"
        
    imgsrc = "/Volumes/USB/segmentationData/instance/Fish4Knowledge/testdataset2/fish/"
    imgsrc = "/Volumes/USB/segmentationData/dataGenerator/foregrounds/modImages/img/"
    imgsrc = "data/datasources/testdataset/fish/"
    imgsrc = src + "fish_image_all/"
    imgs = glob(imgsrc + "*png")

    trainImg, trainMask, valImg, valMask = getDataSplit(imgs, masks, 0.8)

    createCocoData(valMask, valImg, f"{imgsrc}val")
    # createCocoData(trainMask, trainImg, f"{imgsrc}train")
    
    cocosrc = ["data/datasources/testdataset/cocoInfo_Train.json", "data/datasources/testdataset/cocoInfo_Val.json"]

    combineCocoData(cocosrc, "data/datasources/testdataset/cocoInfo_Combined.json")
