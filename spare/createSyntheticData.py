import numpy as np
import cv2
from glob import glob
from random import uniform, shuffle, random
from numpy.lib.utils import info
# import tensorflow as tf
from downloadCocoData import printProgressBar
import os
import albumentations as A
import shutil
# from torch.utils.data import Dataset

'''
This is creating synthetic data based on foreground and background images.
Both can be independently modified (randomally) and combined (randomally)
and the corresponding masks and coco data sets are automatically generated. 

NOTE currently this only works for a single class. 

FUTURE WORK is to make this flexible for multiple classes
    I think a simple way to do this would be to call createSyntheticMedia 
    for each class to be used and the objects of each are just overlayed ontop 
    of each other. The mask for the whole new image is creted at the end of each 
    class and at the very end is added together to create a full mask. 

    This may be the simplest solution, however problems may be that 
    it results in some objects always be ontop of others....?
'''

class augmentImageTF():

    '''
    Object to transform and store information regarding image transformations
    '''

    def __init__(self, img_mask, params):

        '''
        On object initialisation, take the image inputs and modify as necessary

            
            Inputs
        img_mask:       The image/s to modify (image and/or mask)
        params:         Dictionary of parameters and their limits

            Outputs
        self.modImgs:        The augmented image pair (if supplied)
        self.modParam:       The parameters which were used to modify that particular image
        '''
        
        self.img_mask, self.modParam = self.modifyImages(img_mask.copy(), params)

    def modifyImages(self, img_mask, params):

        '''
        Augment the image and apply the mask to the image (if there is 
        one available)
        '''

        modStore = {}
        
        for p in params:
            # get the corresponding function to modify the image 
            if params[p] is not None:
                try:
                    func = eval(f"self.mod_{p}")
                    mod, imgMod = func(img_mask, params.get(p))
                    # perform a check to see if, after tranformations, there
                    # is still information in both the mask and image 
                    if imgMod.get("mask") is not None and imgMod.get("img") is not None:
                        i = imgMod["img"]
                        m = imgMod["mask"]
                        if np.sum(i*m>0) >= np.sum(m>0) * 0.8:
                            img_mask = imgMod
                    else:
                        img_mask = imgMod
                    modStore[p] = mod
                except:
                    modStore[p] = None
                    print(f"mod_{p} did not evaluate for {params.get(p)}")

        # once all transformations are complete, if there is a mask apply it
        img_mask = self.applyMask(img_mask)
        return(img_mask, modStore)
        
    def mod_contrast(self, imgs, param):
        '''
        Modify the contrast of the image within the bounds set by param
        
        The calcuation for contrast adjustment is (x - mean) * param + mean.
        
        Ranges of -1 to 2 are recommended.
        '''

        v = uniform(*param)
        imgs['img'] = tf.image.adjust_contrast(imgs['img'], v).numpy()

        return(v, imgs)

    def mod_brightness(self, imgs, param):
        '''
        Modify the brightness of the image ONLY within the bounds set by param
        
        The calculation for brightness adjustment is x + param.
        
        Ranges of -30 to 30 are recommended.
        '''

        v = uniform(*param)
        imgs['img'] = tf.image.adjust_brightness(imgs['img'], v).numpy()

        return(v, imgs)

    def mod_hue(self, imgs, param):
        '''
        Modify the hue of the image ONLY within the bounds set by param
        
        The calcuation for hue adjustmnet is performed by converting the image
        to HSV and rotating the hue channel (H) by param, then converting back to RGB
        
        Ranges can only be between -1 to 1.
        '''

        v = np.clip(uniform(*param), -1, 1)
        imgs['img'] = tf.image.adjust_hue(imgs['img'], v).numpy()

        return(v, imgs)

    def mod_saturation(self, imgs, param):
        '''
        Modify the hue of the image ONLY within the bounds set by param
        
        The calculation for saturation adjustment is performed by converting
        the images to HSV and multiplying the saturation (S) channel by param and 
        clipping, then converted back to RGB.

        Ranges of -0.5 to 0.5 are recommended.
        '''

        v = uniform(*param)
        imgs['img'] = tf.image.adjust_saturation(imgs['img'], v).numpy()


        return(v, imgs)

    def mod_rotation(self, imgs, param):
        '''
        Modify the rotation of BOTH the image AND mask within the bounds set by param
        
        Maximum ranges of -180 to 180 are sensible but any will work.
        '''

        # NOTE need to take into account the new black space created by the rotation
        # and the loss of the original image around the corners
        angle = uniform(*param)
        shpe = imgs['img'].shape[1::-1]
        image_center = tuple(np.array(shpe) / 2)
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        for i in imgs:
            if imgs.get(i) is not None:
                imgs[i] = cv2.warpAffine(imgs[i], rot_mat, shpe, flags=cv2.INTER_LINEAR)

        return(angle, imgs)

    def mod_xyratio(self, imgs, param):
        '''
        Modify the ratio of the image ONLY within the bounds set by param
        
        Any relative scales are possible.
        '''

        h, w, _ = imgs["img"].shape
        sclh = uniform(*param)
        sclw = uniform(*param)
        h = int(h * sclh)
        w = int(w * sclw)
        for i in imgs:
            if imgs.get(i) is not None:
                imgs[i] = tf.image.resize(imgs[i], [h, w]).numpy().astype(np.uint8)

        return([sclh, sclw], imgs)

    def mod_noise(self, imgs, param):
        '''
        Modify the noise of the image ONLY within the bounds set by param
        
        Recommended max value is 3
        '''

        v = uniform(0, param)
        row,col,ch = imgs["img"].shape
        mean = 0
        gauss = np.random.normal(mean,v,(row,col,ch))
        gauss = gauss.reshape(row,col,ch)
        imgs["img"] = (imgs["img"] + gauss).astype(np.uint8)

        return(v, imgs)

    def mod_shear(self, imgs, param):
        '''
        Modify the shear of the BOTH the image AND the mask within the bounds set by param
        
        Recommended values are between -0.5 to 0.5 but any are possible.
        '''

        v = uniform(*param)
        h, w, _ = imgs["img"].shape
        m = np.float32(
                [[1, v, 0],
                [0, 1  , 0],
                [0, 0  , 1]])
        m[0,2] = -m[0,1] * w/2
        m[1,2] = -m[1,0] * h/2
        for i in imgs:
            if imgs.get(i) is not None:
                imgs[i] = cv2.warpPerspective(imgs[i], m, (h, w)).astype(np.uint8)

        return(v, imgs)

    def mod_horizontalFlip(self, imgs, param):
        '''
        Modify the horizontalFlip of the image within the bounds set by param
        '''

        if random() < 0.5:
            flip = True
            for i in imgs:
                if imgs.get(i) is not None:
                    imgs[i] = tf.image.flip_left_right(imgs[i]).numpy()
        else:
            flip = False

        return(flip, imgs)

    def mod_verticalFlip(self, imgs, param):
        '''
        Modify the verticalFlip of the BOTH image AND mask within the bounds set by param
        '''

        if random() < 0.5:
            flip = True
            for i in imgs:
                if imgs.get(i) is not None:
                    imgs[i] = tf.image.flip_up_down(imgs[i]).numpy()
        else:
            flip = False

        return(flip, imgs)

    def mod_colourBalance(self, imgs, param):
        '''
        Modify the colourBalance of the image within the bounds set by param
        '''

        return(imgs)

    def mod_size(self, imgs, param):
        '''
        Modify the size of the image within the bounds set by param
        
        Any values are possible but values above one are not useful 
        and values too low can result in downsample which is too extreme.
        '''

        h, w, _ = imgs["img"].shape
        scl = uniform(param, 1)
        h = int(h * scl)
        w = int(w * scl)
        for i in imgs:
            if imgs.get(i) is not None:
                imgs[i] = tf.image.resize(imgs[i], [h, w]).numpy().astype(np.uint8)

        return(scl, imgs)

    def mod_lightDirection(self, imgs, param):
        '''
        Modify the lightDirection of the image within the bounds set by param
        '''

        return(imgs)

    def mod_blurring(self, imgs, param):
        '''
        Modify the blurring of the image ONLY within the bounds set by param
        
        Any value is possible but anything about 5 and, especially for 
        small images, features become unrecognisable.

        '''

        v = int(np.round(uniform(0, param)))
        kernel = np.ones((3,3),np.float32)/25
        for _ in range(v):
            imgs["img"] = cv2.filter2D(imgs["img"],-1,kernel)

        return(v, imgs)

    def mod_NLdeform(self, imgs, param):
        '''
        Modify the NLdeform of the image within the bounds set by param
        '''

        return(imgs)

    def mod_occlusion(self, imgs, param):
        '''
        Modify the occlusion of the image within the bounds set by param
        '''

        return(imgs)

    def applyMask(self, imgs):

        '''
        If there a mask available, apply it to the image.
        '''

        if imgs["mask"] is not None:
            imgs["img"] *= ((imgs["mask"] > 127)*1).astype(np.uint8)

        return(imgs)
class augmentImageAL(Dataset):
    def __init__(self, images_filepaths, mask_filespaths = None, transform=None):
        self.images_filepaths = images_filepaths
        self.mask_filespaths = mask_filespaths
        self.transform = transform

    def __len__(self):
        return len(self.images_filepaths)

    def __getitem__(self, idx):
        image_filepath = self.images_filepaths[idx]
        image = cv2.imread(image_filepath)
        name = image_filepath.split("/")[-1].split("_")[0]

        if len(self.mask_filespaths) > 0:
            mask = cv2.imread(self.mask_filespaths[idx])
        else:
            mask = None

        # This can be used to identify various image sources
        if os.path.normpath(image_filepath).split(os.sep)[-2] == "Cat":
            label = 1.0
        else:
            label = 0.0

        if self.transform is not None:
            if mask is None:
                tranformResult = self.transform(image=image)
            else:
                tranformResult = self.transform(image=image, mask = mask)
                mask = tranformResult["mask"]

            image = tranformResult["image"]

        return {"img": image, "mask": mask, "name": name}
class createSyntheticImages():

    '''
    Object to create synthetic images
    '''

    def __init__(self, background, imgs_masks, params):

        '''
        On object initialisation, take the background and apply transformations, 
        as described by params, to overlay the modified foreground images

            Inputs
        background:             Background image 
        imgs_masks:             List of modified foreground foreground image and mask paths
        params:                 Dictionary of parameters to modify the images

            Outputs
        self.synthImage:        Synthetic image combination of the background and foreground objects
        self.synthMask:         Mask of the synthetic foreground objecst and the background. 
                                    NOTE, the values of each pixel corresponds to how many times
                                    invidiaul objects overlap (ie if a pixel = 3 then 3 seperate 
                                    foreground objects have been placed at that point)
        
        '''
        
        self.synthImg, self.synthMask = self.mod_placeForeground(background, imgs_masks, params)

    def mod_placeForeground(self, background, imgs_masks, params):

        '''
        With a list of the fish of interest, place them on the background
        as determined by the parameters
        '''

        clusters = int(uniform(1, params["clusterNo"]))
        backgroundMask = background.copy() * 0

        for c in range(clusters):
            foregroundLoad = self.mod_foreNo(imgs_masks, params["fishNo"])
            background, backgroundMask = self.mod_clusters(background, foregroundLoad, backgroundMask, params["clusterDensity"])

        return(background, backgroundMask)

    def mod_foreNo(self, imgs_masks, param):
        '''
        Get the param number of randomly selected foreground image.

        Each image/mask pair is only used once per function call.
        '''

        # There has to be at least one image captured
        if param is None:
            param = 1

        fishNo = int(uniform(*param))
        fishID = np.arange(len(imgs_masks))
        shuffle(fishID)
        imgs_masksSelect = [imgs_masks[i] for i in fishID[:fishNo]]

        fishLoad = []
        for i in imgs_masksSelect:
            fishLoad.append({"img": cv2.imread(i["img"]), "mask": cv2.imread(i["mask"])})

        return(fishLoad)

    def mod_clusters(self, background, imgs_masks, backgroundMask, param):
        '''
        Modify the clustering of the image within the bounds set by param
        
        param is from 0 to 1 where a value of 0 indicates that fish are 
        placed within a 0% size of the background and 1 within the full size 
        of the image.

        Recommended values are 0.3 to 1
        '''

        # adjust the cluster size for the x and y axis independently
        cX = uniform(*param)
        cY = uniform(*param)
        x, y, _ = background.shape

        # create the cluster area in which the foreground objects will be randomally placed
        cluster = np.zeros([int(x*cX), int(y*cY), 3]).astype(np.uint8)
        clusterMask = cluster.copy()
        xs, ys, _ = cluster.shape
        for i in imgs_masks:
            img = i["img"]
            mask = ((i["mask"] > 200)*1).astype(np.uint8)       # ensure mask is binary
            xf, yf, _ = img.shape

            # if the target foreground is larger than the possible cluster 
            # then just continue
            if xs < xf or ys < yf:
                continue

            xp = int(uniform(0, xs - xf - 1))
            yp = int(uniform(0, ys - yf - 1))

            cluster[xp:xp+xf, yp:yp+yf, :] *= 1-mask
            clusterMask[xp:xp+xf, yp:yp+yf, :] += mask
            cluster[xp:xp+xf, yp:yp+yf, :] += img * mask
            
        xb = int(uniform(0, x-xs))
        yb = int(uniform(0, y-ys))
        
        backgroundMask[xb:xb+xs, yb:yb+ys, :] += clusterMask
        clusterMaskBinary = ((clusterMask == 0) * 1).astype(np.uint8)
        background[xb:xb+xs, yb:yb+ys, :] *= clusterMaskBinary
        background[xb:xb+xs, yb:yb+ys, :] += cluster 

        return(background, backgroundMask)

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

def getMediaTF(mediasources, imgDest, synthNo = None, perSynthNo = 1, frameNo = 30, loadImgs = False, param = None):

    '''
    Get the foreground images and their MASKS and modifiy them for the given parameters

        Inputs
    mediasources:       Dictionary of the foreground and masks supplied. 
    imgDest:            Destination of the media extracted, both raw and modified.
                            NOTE the directories and sub-directories are automatically created.
    synthNo:            The max number of synthetic images to generate from each data source. 
                            If False then process all media presented
    perSynthNo:         The number of synthetic images to generate PER supplied **source**
                            If synthNo specified then the perSynthNo for the specified max number of images per source
    frameNo:            For videos, the rate of frame capture (ie 30 means only every 30 frames save an image)
    loadImgs:           If True, this will remove remove any pre-existing directory and will extract
                            and save the images from the media. This needs to be set to True the first time 
                            data is processed but can/should be set to False for any subsequent processing.
    param:              Dictioary of the parameters to modify the background. For every parameter see the 
                            corresponding function for acceptable ranges. If set to None will augment (at all or for the 
                            specified parameter).
    '''         

    print(f"\n\nInformation is being created at {imgDest}")

    # NOTE have function to create directories
    imgDestOrig = imgDest + "origImages/"
    imgDestMod = imgDest + "modImages/"

    for name in mediasources:
        media = mediasources[name]

        if loadImgs:
            # if loadImgs is True, remove the ENTIRE directory and populate with new data
            dirMaker(imgDest, True)     
            for m in media:
                loadImg(media[m], name, f"{imgDestOrig}{m}/", frameNo, synthNo)

        augmentImagesTF([imgDestOrig + f"mask/{name}*.jpg", imgDestOrig + f"img/{name}*.jpg"], imgDestMod, synthNo, perSynthNo, param)

def getMediaAL(mediasources, imgDest, synthNo = None, perSynthNo = 1, frameNo = 30, loadImgs = False, param = None):

    '''
    Get the foreground images and their MASKS and modifiy them for the given parameters

        Inputs
    mediasources:       Dictionary of the foreground and masks supplied. 
    imgDest:            Destination of the media extracted, both raw and modified.
                            NOTE the directories and sub-directories are automatically created.
    synthNo:            The max number of synthetic images to generate from each data source. 
                            If False then process all media presented
    perSynthNo:         The number of synthetic images to generate PER supplied **source**
                            If synthNo specified then the perSynthNo for the specified max number of images per source
    frameNo:            For videos, the rate of frame capture (ie 30 means only every 30 frames save an image)
    loadImgs:           If True, this will remove remove any pre-existing directory and will extract
                            and save the images from the media. This needs to be set to True the first time 
                            data is processed but can/should be set to False for any subsequent processing.
    param:              Dictioary of the parameters to modify the background. For every parameter see the 
                            corresponding function for acceptable ranges. If set to None will augment (at all or for the 
                            specified parameter).
    '''         

    print(f"\n\nInformation is being created at {imgDest}")

    # NOTE have function to create directories
    imgDestOrig = imgDest + "origImages/"
    imgDestMod = imgDest + "modImages/"

    # if loadImgs is True, remove the ENTIRE directory and populate with new data
    if loadImgs:
        dirMaker(imgDest, True)     
        for name in mediasources:
            media = mediasources[name]

            for m in media:
                loadImg(media[m], name, f"{imgDestOrig}{m}/", frameNo, synthNo)

    augmentImagesAL([imgDestOrig + f"mask/*.jpg", imgDestOrig + f"img/*.jpg"], imgDestMod, perSynthNo, param)

def loadImg(media, mediaName, imageDest, frameNo, synthNo):

    '''
    From the dictionary which is specifying the media to load, convert all 
    input into numpy arrays
    '''

    # it will either be an image, images or a video
    # VideoCapture apparantely can process images as well???

    dirMaker(imageDest)

    # if the synth number is not specified, process all
    if synthNo is None:
        synthNo = np.inf

    if media.find("*")>-1:
        imgs = sorted(glob(media))
        if synthNo: imgs = imgs[:synthNo]
        for n, i in enumerate(imgs):
            os.system(f"cp {i} {imageDest}{mediaName}_{n}.jpg")

    else:
        vidcap = cv2.VideoCapture(media)
        success = True
        c = 0
        while True:
            success,img = vidcap.read()
            if success and c%frameNo == 0:
                cv2.imwrite(f"{imageDest}{mediaName}_{c}.jpg", img)
            elif success is False or c > (synthNo-1)*frameNo: 
                break
            c+=1

def imgMaskList(imgs, masks):

    ''' 
    Create a list which combines corresponding images and masks 
    '''

    infoAll = []

    # if we have single image input 
    if type(masks) is type(np.array([])):
        infoAll.append({"img": imgs, "mask": masks})

    # if there is no mask info then we are only processing the images
    elif len(masks) == 0:
        for i in imgs:
            infoAll.append({"img": i, "mask": None})

    # if we have both mask and images, process together
    else:
        for i, m in zip(imgs, masks):
            infoAll.append({"img": i, "mask": m})

    return(infoAll)

def augmentImagesTF(imgsrc, imgdest, synthNo, perSynthNo, params):

    '''
    Take an image and randomally modify it given the paraameters provided with 
    tensorflow and manual augmentations
    '''

    dirMaker(imgdest + "mask/")
    dirMaker(imgdest + "img/")

    infoAll = []
    # if input is a path
    if type(imgsrc) is list:
        masks, imgs = map(glob, imgsrc)

    # if input is a list of loaded images
    else:
        masks, imgs = imgsrc

    infoAll = imgMaskList(imgs, masks)

    # use a random selection of the images if there is a limit on the number
    # of images to use
    if synthNo:
        imgR = np.arange(len(imgs))
        # shuffle(imgR)     # NOTE re-activate for evaluation...
        infoAll = [infoAll[r] for r in imgR[:synthNo]]

    for n, img_mask in enumerate(infoAll):
        name = img_mask["img"].split("/")[-1].split(".")[0]
        media = {}
        media["img"] = cv2.imread(img_mask["img"])
        media["mask"] = cv2.imread(img_mask["mask"])

        for nb in range(perSynthNo):
            augResults = augmentImageTF(media, params)
            modParams = augResults.modParam
            modImg = augResults.img_mask
            for m in modImg:
                if modImg.get(m) is not None:
                    cv2.imwrite(f"{imgdest}{m}/{name}_{nb}.jpg", modImg[m])
            printProgressBar(nb + n*perSynthNo+1, len(infoAll) * perSynthNo, f"{name} processing", length = 10)

def augmentImagesAL(imgsrc, imgdest, perSynthNo, params):

    '''
    Take an image and randomally augment it using albumentations
    '''

    dirMaker(imgdest + "mask/")
    dirMaker(imgdest + "img/")

    if type(imgsrc) is list:
        masks, imgs = map(glob, imgsrc)

    # if input is a list of loaded images
    else:
        masks, imgs = imgsrc

    # use a random selection of the images if there is a limit on the number
    # of images to use
    
    augmenter = augmentImageAL(imgs, masks, params)
    augLen = augmenter.__len__()
    for n in range(augmenter.__len__()):
        for p in range(perSynthNo):
            modImgs = augmenter[n]  
            name = modImgs["name"]

            imgStore = []
            for m in modImgs:
                modImg = modImgs[m]
                if type(modImg) is type(np.array([])):
                    cv2.imwrite(f"{imgdest}{m}/{name}_{p}.jpg", modImg)
                    imgStore.append(modImg)
            cv2.imshow("IMG", np.hstack(imgStore)); cv2.waitKey(0)
            # modImg = augResults.img_mask
            
            printProgressBar(p + n*perSynthNo+1, augLen * perSynthNo, "Augmenting", length = 10)

def createSyntheticMedia(imgsrc, params, dest, synthNoBck = 1, synthNoFore = 1):

    '''
    Combine the modified foregrounds and backgrounds to create new
    synthetic data.

        Inputs
    imgsrc:         Home path to the directory containing all the images
    forePath:       Home path to the directory containing foreground images and masks
    params:          Dictionary of the parameters to generate/modify the synthetic data
    dest:           Destination to save the synthetic data
    synthNoBck:     Number of synthetic images to generate PER background
    synthNoFore:    Number of synthetic images to generate PER foreground

    '''

    # get the hardcoded paths of all the data
    bcksrc = imgsrc + "backgrounds/modImages/img/"
    foresrcImg = imgsrc + "foregrounds/modImages/img/"
    foresrcMask = imgsrc + "foregrounds/modImages/mask/"

    imgDest = dest + "imgs/"
    maskDest = dest + "masks/"
    dirMaker(imgDest)
    dirMaker(maskDest)

    backgroundImgs = sorted(glob(bcksrc + "*.jpg"))
    foreImgs = sorted(glob(foresrcImg + "*.jpg"))
    foreMasks = sorted(glob(foresrcMask + "*.jpg"))

    imgs_masks = imgMaskList(foreImgs, foreMasks)

    for n, b in enumerate(backgroundImgs * synthNoBck):
        backImg = cv2.imread(b)
        synthImgs = createSyntheticImages(backImg, imgs_masks, params)

        synthImg = synthImgs.synthImg
        synthMask = synthImgs.synthMask

        synthMask = ((synthMask>0)*255).astype(np.uint8)

        cv2.imwrite(f"{imgDest}{n}.jpg", synthImg)
        cv2.imwrite(f"{maskDest}{n}.jpg", synthMask)

        printProgressBar(n+1, len(backgroundImgs) * synthNoBck, "SynthData", length=10)


    # not sure i need this??
    # augmentImages([imgMod, maskMod], dest, 1, synthMods, param)

if __name__ == "__main__":

    # dictionary containing the foreground directories to the images AND their masks
    foreground = {
        "fish4knowledge": {"img": "/Volumes/USB/segmentationData/instance/Fish4Knowledge/fish_image/*/*", 
        "mask": "/Volumes/USB/segmentationData/instance/Fish4Knowledge/mask_image/*/*"}
    }

    # dictionary containing the background media
    background = {
        "waterimg": {"img": "/Volumes/USB/segmentationData/dataGenerator/sources/water.jpeg"},
        "rivervid": {"img": "/Volumes/USB/segmentationData/dataGenerator/sources/RiverFish.mp4"}
    }
    
    imageDest = "/Volumes/USB/segmentationData/dataGenerator/"

    foreground = {
        "fish4knowledge": {"img": "/Users/jonathanreshef/Downloads/fish_23/*png", "mask": "/Users/jonathanreshef/Downloads/mask_23/*png"}
    }

    background = {
        "waterimg": {"img": "/Users/jonathanreshef/Downloads/water.jpeg"},
        "cat": {"img": "testingAugmentation/cat.jpg"},
        "rivervid": {"img": "/Volumes/USB/segmentationData/dataGenerator/sources/RiverFish.mp4"}
    }

    imageDest = "testingAugmentation/"

    '''
    NOTE - parameters to adjust:
    Both:
        contrast
        rotation
        ratio
        noise
        shear
        horizontalFlip
        verticalFlip
        colourBalance (mostly for the background, fish colour is more consistent)
        lightDirection

    Foreground specific:
        size
        blurring
        NLdeform
        occlusion

    Background specific:
        overlay of non-target but non-background objects (such as nets, docks etc.)
    
    '''
    paramForegroundTF = {
        "contrast": None,                   # variance of contrast
        "brightness": [-0.2, 0.2],          # min and max brightness
        "hue": None, # [-0.2, 0.2],                 # min and max adjust hue
        "saturation": [0, 0.5],          # min and max saturation
        "rotation": [-180, 180],            # min and max rotation from intital position
        "xyratio": [0.8, 1.2],                # min to max x and y distortion
        "noise": None,                         # maximum noise added to images (mean = 0, value is sigma around the mean)
        "shear": [-0.1, 0.1],                # min and max x and y shear directions
        "horizontalFlip": True,             # allow horizontal flipping (50/50 chance)
        "verticalFlip": True,               # allow vertical flipping (50/50 chance)
        "colourBalance": None,              # maximum variance of each colour channel (maybe multiples of the RGB channels???)
        "size": 0.5,                        # min rescale of image (1 = original size)
        "lightDirection": None,             # TBC, could be OTT...
        "blurring": None,                      # maximum number of blur iterations (5x5 filter)
        "NLdeform": None,                   # the maximum % of horizontal distance change in a distortion
    }

    paramBackgroundTF = {
        "contrast": None,                   # variance of contrast
        "brightness": [-0.2, 0.2],          # min and max brightness
        "hue": None,                 # adjust hue
        "saturation": [0, 0.5],          # min and max saturation
        "rotation": None,                   # rotation from intital position
        "xyratio": [0.5, 1.5],                # x and y min to max distortion
        "noise": None,                         # maximum noise added to images (mean = 0, value is sigma around the mean)
        "shear": None,                # x and y min and max shear directions
        "horizontalFlip": True,             # allow horizontal flipping
        "verticalFlip": None,              # allow vertical flipping
        "colourBalance": None,              # maximum variance of each colour channel (maybe multiples of the RGB channels???)
        "size": None,                        # min rescale of image (1 = original size)
    }

    # create randomally generate images
    # getMediaTF(foreground, imageDest + "foregrounds/", synthNo = None, perSynthNo = 3, loadImgs = True, param = paramForegroundTF)
    # getMediaTF(background, imageDest + "backgrounds/", synthNo = None, perSynthNo = 3, loadImgs = True, param = paramBackgroundTF)


    params = A.Compose([
        # A.OpticalDistortion(distort_limit=1, shift_limit=1)
        # A.ElasticTransform(10, 200, 200)
        A.RGBShift()

    ])


    getMediaAL(foreground, imageDest + "foregrounds/", synthNo = 5, perSynthNo = 10, loadImgs=True, param = params)
    getMediaAL(background, imageDest + "backgrounds/", synthNo = 5, perSynthNo = 1, loadImgs=True, param = params)


    paramSynthetic = {
        "overlap": None,
        "occlusion": None,
        "fishNo": [10, 20],
        "clusterDensity": [0.1, 0.3],
        "clusterNo": 6
    }

    createSyntheticMedia(imageDest, paramSynthetic, imageDest + "syntheticData/", 100)