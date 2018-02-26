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
                lines = openedFile.read().splitlines()
                npLines = np.asarray(lines)
                oneImageFeatures = npLines[2::8]
                for index in range(3,9):
                    oneImageFeatures = np.core.defchararray.add(oneImageFeatures, npLines[index::8])
                oneImageFeatures = np.asarray(np.core.defchararray.split(np.array([''.join(oneImageFeatures)]))[0]).astype(np.float)
                oneImageFeatures = oneImageFeatures.reshape(oneImageFeatures.shape[0]/128, 128)
                print oneImageFeatures[0,:]
                print oneImageFeatures[-1,:]
                raw_input()
                allImagesFeatures.append(oneImageFeatures)
                print "progress: {} / 150".format(numOfFiles)

def main():
    preprocess()

if __name__ == '__main__':
    startTime = time.time()
    main()
    print  "processing time: {} seconds".format(time.time()-startTime)
