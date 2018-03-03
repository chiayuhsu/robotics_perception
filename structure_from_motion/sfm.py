#!/usr/bin/env python

import numpy as np
import os
import time
import cv2
import pickle
import csv
from joblib import Parallel, delayed
import multiprocessing


siftPath = "./sift"
NOI = 150 # number of images
allImagesFeatures = []
fileName = []
resultScore = np.zeros((NOI,NOI))

def preprocess():
    siftFiles = os.listdir(siftPath)
    numOfFiles = 0
   
    for f in siftFiles:
        if f.endswith(".sift"):
            fileName.append(f)
            numOfFiles += 1
            with open(os.path.join(siftPath, f), "r") as openedFile:
                lines = openedFile.read().splitlines()
                npLines = np.asarray(lines)
                oneImageFeatures = npLines[2::8]
                for index in range(3,9):
                    oneImageFeatures = np.core.defchararray.add(oneImageFeatures, npLines[index::8])
                oneImageFeatures = np.asarray(np.core.defchararray.split(np.array([''.join(oneImageFeatures)]))[0]).astype(np.float32)
                oneImageFeatures = oneImageFeatures.reshape(oneImageFeatures.shape[0]/128, 128)
                allImagesFeatures.append(oneImageFeatures)
                print "progress: {} / {}".format(numOfFiles, NOI)

def matching(i):
    bf = cv2.BFMatcher()
   # for i in range(len(allImagesFeatures)):
    matchPercent = [1]
    print "Matching {}th image:".format(i+1)
    for j in range(i+1, NOI):
        #st = time.time()
        matches = bf.knnMatch(allImagesFeatures[i], allImagesFeatures[j], k=2)
        #print "{}".format(time.time()-st)             
            
        goodMatch = 0
        for m, n in matches:
            if m.distance < 0.75*n.distance:
                goodMatch += 1
        matchPercent.append(float(goodMatch) / len(matches))
        #print "progress: {} / {}".format(j-i, NOI-i-1)
    #resultScore[i,i:] = np.asarray(matchPercent)
    #resultScore[i:,i] = np.asarray(matchPercent)
    #print resultScore
    print "Finished matching {}th image:".format(i+1)
    return np.asarray(matchPercent)

def savetofile():
    with open('output.csv', 'wb') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(['ImageName', 'FirstMatchImage', 'FirstMatchScore', 'SecondMatchImage', 'SecondMatchScore','ThirdMatchImage', 'ThirdMatchScore','FourthMatchImage', 'FourthMatchScore','FifthMatchImage', 'FifthMatchScore'])
        for i in range(NOI):
            orderIndex = np.flip(np.argsort(resultScore[i,:]), axis=0)
            score = np.round(resultScore[i,orderIndex[0:6]] * 100).astype(int)
            writeList = [fileName[i][:-5]]
            for j in range(5):
                writeList.append(fileName[orderIndex[j+1]][:-5])
                writeList.append(score[j+1])
            csvwriter.writerow(writeList) 

def main():
    print "Loading feature files..."
    startTime = time.time()
    preprocess()
    print  "Processing time: {} seconds".format(time.time()-startTime)
    print "Matching images..."
    startTime = time.time()
    numCores = multiprocessing.cpu_count()
    resultList = Parallel(n_jobs=numCores)(delayed(matching)(i) for i in range(len(allImagesFeatures)))
    for i in range(len(resultList)):
        resultScore[i,i:] = resultList[i]
        resultScore[i:,i] = resultList[i]
     
    #matching()
    print  "Processing time: {} seconds".format(time.time()-startTime)
    savetofile()    


if __name__ == '__main__':
    main()
