#!/usr/bin/env python

import numpy as np
import os
import time

siftPath = "./sift"
allImagesFeatures = []

def preprocess():
    siftFiles = os.listdir(siftPath)
    numOfFiles = 0
   
    for f in siftFiles:
        if f.endswith(".sift"):
            numOfFiles += 1
            with open(os.path.join(siftPath, f), "r") as openedFile:
                oneImageFeatures = np.array([])
                oneSpotFeature = []
 
                for line in openedFile:
                    lineSplit = line.split()
                    if len(lineSplit) == 20:
                        oneSpotFeature += lineSplit
                    elif len(lineSplit) == 8:
                        oneSpotFeature += lineSplit
                        oneImageFeatures = np.append(oneImageFeatures, np.asarray(oneSpotFeature), axis=0)
                        oneSpotFeature = []

                oneImageFeatures = oneImageFeatures.reshape(oneImageFeatures.shape[0]/128, 128).astype(np.float)
                allImagesFeatures.append(oneImageFeatures)
                print "progress: {} / 150".format(numOfFiles)

def main():
    preprocess()

if __name__ == '__main__':
    startTime = time.time()
    main()
    print  "processing time: {} seconds".format(time.time()-startTime)
