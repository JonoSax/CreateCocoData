'''
This script will get data from cloud sources or process images and videos
into a standard format
'''

from os import error
import cv2
import numpy as np
from glob import glob
from skimage.registration import phase_cross_correlation as pcc
from cocoDataStructure.utilities import dirMaker, printProgressBar


def sigFrame(err, errStore, minFrameDist):

    '''
    Determine if an error value is a statistically signficant change
    '''

    if len(errStore) < minFrameDist:
        return(False)

    diff = np.diff(errStore)
    std = np.std(diff)
    me = np.mean(diff)

    errDiff = err - errStore[-1]

    if errDiff > me + std * 2 or errDiff < me - std * 2:
        return(True)
    else:
        return(False)


def extractFrames(vidPath, dest = None, minFrameDist = 10):

    '''
    From a video, extract frames which are sufficiently different
    '''

    vidName = vidPath.split("/")[-1]

    cap = cv2.VideoCapture(vidPath)

    if (cap.isOpened()== False): 
        print("Error opening video stream or file")
        return

    if dest is None:
        dest = f'{"/".join(vidPath.split("/")[:-1])}/extract/'

    dirMaker(dest)

    x = cap.get(cv2.CAP_PROP_FRAME_WIDTH)   
    y = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)  
    fps = cap.get(cv2.CAP_PROP_FPS)           
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)   

    frame0 = (np.random.random([300,300]) * 255).astype(np.uint8)
    
    n = 0
    errStore = []

    # save frames which are significantly different from other frames
    while(cap.isOpened()):
        printProgressBar(n, frames, "Frames processed", length=20)
        ret, frame = cap.read()

        # downsample and convert to grayscale to speed up the correlation comparsions
        frame1 =  cv2.blur(cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (300, 300)), (5, 5))

        if ret == True:
            shift, err, phasedif = pcc(frame0, frame1, upsample_factor=0.2)

            sig = sigFrame(err, errStore, minFrameDist)

            # if there is a signficant difference in frames re-assign the 
            # frame store
            if (sig or err > 0.2) and len(errStore) > minFrameDist: # (err > 0.2 or np.isnan(err)) and np.sum(frame)/frame.size > 20:
                cv2.imwrite(f"{dest}{vidName}_{n}.jpg", frame)
                errStore = []
                # cv2.imshow("imgs", np.hstack([frame0, frame1])); cv2.waitKey(0)
                frame0 = frame1
            else:
                errStore.append(err)

        else: 
            break

        n += 1

    cap.release()




if __name__ == "__main__":

    src = "/media/boxfish/USB/KINGSTON/testVids/"
    videos = sorted(glob(src + "*.mp4"))

    for v in videos:

        extractFrames(v)
