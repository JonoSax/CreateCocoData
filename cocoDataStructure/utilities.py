'''
Functions which are used across multiple different scripts
'''

import json
from glob import glob
import numpy as np
import shutil
import os

def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()

def createIDDict(targetdict, keytype, classtype, pathsrc = None):

    '''
    Create a dictionary which extracts information from within a list of dictionaries
    
    Inputs:
        targetdict: list of dictionaries
        keytype:   the dictionary key of the information to be used for the new dictionary key
        classtype: the dictionary key of the information to be associated with the new dictionary value
                        if classtype == *, then all dictionary information is appended
        pathsrc:   additional information to be used for the new dictionary key

    Outputs:
        imgDict:  dictionary with key type as keys and class type as the information 
    '''
    

    imgDict = {}

    for i in targetdict:
        key = i[keytype]
        if imgDict.get(key) is None:
            imgDict[key] = []
        if classtype == "*":
            imgDict[key].append(i)
        elif pathsrc is None:
            imgDict[key].append(i[classtype])
        else:
            imgDict[key].append(pathsrc + str(i[classtype]))

    return(imgDict)

def associateImageID(src):

    print(f"Associating {src.split('/')[-2]} id")

    imgs = sorted(glob(src + "images/**/*"))

    imgNames = [i.split("/")[-1] for i in imgs]

    imgDict = {}
    for n, i in enumerate(imgNames):  
        imgDict[i] = n

    json.dump(imgDict, open(src + "imgDict.json", "w"))

    print(f"    Finished {src.split('/')[-2]}")

    return(imgDict)

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

def PolyArea(x,y):

    '''
    Shoe lace formula
    https://stackoverflow.com/questions/24467972/calculate-area-of-polygon-given-x-y-coordinates
    '''
    
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))
            
def dirMaker(dir, remove = False):

    '''
    creates directories (including sub-directories)

    Input:    \n
    (dir), path to be made
    (remove), if true, if the directory already exists remove

    Output:   \n
    (), all sub-directories necessary are created\n
    (made), boolean whether this is the first time the directory has been made
    '''

    def make():
        dirToMake = ""
        for d in range(dir.count("/")):
            dirToMake += str(dirSplit[d] + "/")
            try:
                os.mkdir(dirToMake)
                madeNew = True
            except:
                madeNew = False

        return(madeNew)

    # ensure that the exact directory being specified exists, if not create it
    dirSplit = dir.split("/")

    madeNew = make()

    # if the directory exists and the user want to create a clean directory, remove the 
    # dir and create a new one
    if madeNew == False and remove == True:
        shutil.rmtree(dir)
        madeNew = make()
    
    return(madeNew)
