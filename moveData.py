from glob import glob
import os
fishsrc = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/fish/"
fishdest = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/fish_all/"
masksrc = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/mask/"
maskdest = "/Volumes/WorkStorage/BoxFish/dataStore/fishData/YOLO_data/Ulucan/mask_all/"

fishtypes = sorted(glob(fishsrc + "*"))
# masktypes = sorted(glob(masksrc + "*"))


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
