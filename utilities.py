'''
Functions which are used across multiple different scripts
'''

import json
from glob import glob

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

    imgs = sorted(glob(src + "masks/**/*"))

    imgNames = [i.split("/")[-1] for i in imgs]

    imgDict = {}
    for n, i in enumerate(imgNames):  
        imgDict[i] = n

    json.dump(imgDict, open(src + "imgDict.json", "w"))

    return(imgDict)

