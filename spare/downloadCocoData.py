from multiprocessing import Pool, set_start_method
import os
import json
from itertools import repeat
from utilities import printProgressBar, createIDDict

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

    

