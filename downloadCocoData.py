from multiprocessing import Pool, set_start_method
import os
import json
from overalySegments import createIDDict
from itertools import repeat

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

def getImage(dest, i):
    imgurl = i["coco_url"]
    imgdest = dest + imgurl.split("/")[-1]
    os.system(f"wget {imgurl} -O {imgdest} >/dev/null 2>&1")

def downloadImages(dest, segments, parallel = True, limit = False):

    '''
    Form the supplied cocourl, downlaod the corresponding image and create the 
    necessary coco structure to accompany
    '''

    annotations = segments["annotations"]
    images = segments["images"][100:]

    # if creating only a subset of the full coco info, extract only the 
    # necessary images and corresponding annotations
    annotationsLim = []
    imagesLim = []
    if limit:
        annoAll = createIDDict(annotations, "image_id", "*", pathsrc = "")
        annotations = []
        for i in images:
            try:
                imgAnno = annoAll[i["id"]]
                for a in imgAnno:
                    annotationsLim.append(a)
                imagesLim.append(i)
                if len(imagesLim) == limit:
                    images = imagesLim
                    annotations = annotationsLim
                    break
            except:
                continue

    print("Starting image downloads")
    if parallel:
        with Pool(processes = os.cpu_count() - 1) as pool:
            pool.starmap(getImage, zip(repeat(dest), images, ))

    else:
        # serialised
        for n, i in enumerate(images):
            getImage(dest, i)
            printProgressBar(n, len(images)-1, "Imgs downloaded", length = 20)


    print(f"Done downloading into {dest}")

    newSegments = segments.copy()
    newSegments["images"] = images
    newSegments["annotations"] = annotations

    return(newSegments)

if __name__ == "__main__":

    set_start_method("spawn")
    imgsrc = "data/datasources/coco/val17_images/"
    segsrc = "data/datasources/coco/coco_train17All.json"
    dest = "data/datasources/coco/val17.json"
    segFile = open(segsrc)
    segments = json.load(segFile)

    newSegment = downloadImages(imgsrc, segments, limit = 20)

    newSegmentJson = open(dest, "w")
    json.dump(newSegment, newSegmentJson)
    newSegmentJson.close()

    

