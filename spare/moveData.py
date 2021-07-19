from glob import glob
import os
from utilities import printProgressBar


fishsrc = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/fish/"
fishdest = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/fish_all/"
masksrc = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/mask/"
maskdest = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/mask_all/"

fishtypes = sorted(glob(fishsrc + "*"))
# masktypes = sorted(glob(masksrc + "*"))

for f in fishtypes: #, masktypes):
    species = f.split("/")[-1].replace(" ", "")
    fishimgs = sorted((glob(f + "/*.png")))
    # maskimgs = sorted((glob(m + "/*.png")))
    for n, i in enumerate(fishimgs): #, maskimgs)):
        i = i.replace(" ", "\ ")
        printProgressBar(n, len(fishimgs) - 1, species, length=20)
        newFishName = f"{fishdest}{species}_{n}.png"
        # newMaskName = f"{maskdest}{species}_{n}.png"
        os.system(f"cp {i} {newFishName}")
        # os.system(f"cp {m} {newMaskName}")
