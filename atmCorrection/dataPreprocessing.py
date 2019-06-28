# =============================================================================
#     
#   WorldView 3 Atmospheric Correction Neural Network Preprocessing Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Prepares a WV3 image for training using NDVI selection criterion
# 
#   TODO
#
#
# =============================================================================

from __future__ import division
import geoio
import gdal
import numpy as np
from joblib import dump
from datetime import datetime
import matplotlib
import scipy

# =============================================================================
# User Specified Values
# =============================================================================
inputImgFileName = '19MAY15040556-M1BS-503159090010_01_P001.NTF'
inputImgFileName = 'trainImage2.NTF'

outputImgFileName = 'step3.nitf'
outputImgFileName = 'outImage2.NTF'

metadataFileName = '/home/grl/Documents/Trevor/ImageClassification/TrainingTool/atmCorrection/19MAY15040556-M1BS-503159090010_01_P001.NTF'
metadataFileName = '/home/grl/Documents/Trevor/ImageClassification/TrainingTool/atmCorrection/trainImage2.NTF'

trainDataName = 'trainSet1'
truthDataName = 'truthSet1'

maxDimSize = 5000

sampleSize = 500000
lowHighCount = int(sampleSize/5)
medCount = int(sampleSize/10*3)

# =============================================================================
# Read in data and attributes
# =============================================================================
inputImg = gdal.Open(inputImgFileName)
outputImg = gdal.Open(outputImgFileName)
metadataFull = geoio.DGImage(metadataFileName)
metadata = metadataFull.meta
metadataSun= metadataFull.meta_dg.IMD.IMAGE
sunData = np.array([metadataSun.MEANSUNAZ, metadataSun.MEANSUNEL, metadataSun.MEANSATAZ, metadataSun.MEANSATEL, metadataSun.MEANOFFNADIRVIEWANGLE])

inFeatNum = inputImg.RasterCount
outFeatNum = outputImg.RasterCount
imageXSize = inputImg.RasterXSize
if imageXSize > maxDimSize:
    imageXSize = maxDimSize
imageYSize = inputImg.RasterYSize
if imageYSize > maxDimSize:
    imageYSize = maxDimSize
imageSize = imageXSize*imageYSize

inputArray = inputImg.ReadAsArray(0,0,imageXSize,imageYSize).astype('float32').reshape(inFeatNum,imageSize).transpose()
outputArray = outputImg.ReadAsArray(0,0,imageXSize,imageYSize).astype('float32').reshape(inFeatNum,imageSize).transpose()

# =============================================================================
# Compute NDVI values for selection
# =============================================================================
redBand = inputArray[:,2]
nirBand = inputArray[:,3]

ndvi = (nirBand-redBand)/(nirBand+redBand)
matplotlib.pyplot.hist(ndvi,bins=500)

median = scipy.median(ndvi)
q1 = scipy.median(ndvi[np.where(ndvi<median)])
q3 = scipy.median(ndvi[np.where(ndvi>median)])

highVegInd = np.where(ndvi>q3)[0]
lowVegInd = np.where(ndvi<q1)[0]
medHighVegInd = np.setdiff1d(np.where(ndvi>median)[0],highVegInd)
medLowVegInd = np.setdiff1d(np.where(ndvi>q1)[0],np.where(ndvi>median)[0])

# =============================================================================
# Select values for training
# =============================================================================

trainData = np.zeros((sampleSize,inFeatNum))
truthData = np.zeros((sampleSize,outFeatNum))

trainData[:lowHighCount,:] = inputArray[np.random.choice(lowVegInd,lowHighCount),:]
trainData[lowHighCount:lowHighCount+medCount,:] = inputArray[np.random.choice(medLowVegInd,medCount),:]
trainData[lowHighCount+medCount:lowHighCount+2*medCount,:] = inputArray[np.random.choice(medHighVegInd,medCount),:]
trainData[lowHighCount+2*medCount:,:] = inputArray[np.random.choice(highVegInd,lowHighCount),:]

truthData[:lowHighCount,:] = outputArray[np.random.choice(lowVegInd,lowHighCount),:]
truthData[lowHighCount:lowHighCount+medCount,:] = outputArray[np.random.choice(medLowVegInd,medCount),:]
truthData[lowHighCount+medCount:lowHighCount+2*medCount,:] = outputArray[np.random.choice(medHighVegInd,medCount),:]
truthData[lowHighCount+2*medCount:,:] = outputArray[np.random.choice(highVegInd,lowHighCount),:]

# =============================================================================
# Radiometric correction
# =============================================================================
gainOffsetData = np.array([[0.94,-5.809],[0.938,-4.996],[0.964,-3.021],[0.961,-5.522]])
for i in range(inFeatNum):
    scalar = gainOffsetData[i,0]*metadata.abscalfactor[i]/metadata.effbandwidth[i]/10
    trainData[:,i] = trainData[:,i]*scalar+gainOffsetData[i,1]


# =============================================================================
# Additional Features (Haze Optimize Transform)
# =============================================================================
tmpArray = np.zeros((sampleSize,inFeatNum+6))
for i in range(sampleSize):
    vec = inputArray[i,:]
    m = vec[0]/vec[3]
    theta = np.arctan(m)
    hot = vec[0]*np.sin(theta) - vec[1]*np.cos(theta)
    tmpArray[i,:inFeatNum] = trainData[i,:]
    tmpArray[i,inFeatNum+1] = hot
    tmpArray[i,inFeatNum+1:] = sunData

trainData = tmpArray

tmpArray = None
inFeatNum +=6
print datetime.now()



# =============================================================================
# Save training data to be used later (and also output subset)
# =============================================================================
dump(trainData,trainDataName)
dump(truthData,truthDataName)