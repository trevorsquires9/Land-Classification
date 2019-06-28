# =============================================================================
#     
#   WorldView 3 Atmospheric Correction Neural Network Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Trains a neural network to learn how to perform atmospheric correction
#       on WorldView 3 satellite images
# 
#   TODO
#   - Test model
#   - Write successfully
#
# =============================================================================


import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import svm
from sklearn import linear_model
from sklearn.neural_network import MLPRegressor
import pandas as pd
from Tkinter import *
import gdal
import geoio
from datetime import datetime
from osgeo import gdal_array
from joblib import dump,load
import pickle


# =============================================================================
# Define functions for GUI
# =============================================================================
def grabFileNames():
    global fileName,metadata,outFileName
    fileName = fileName.get()
    metadata = metadata.get()
    outFileName = outFileName.get()
    master.destroy()

# =============================================================================
# Define the GUI used for obtaining the data and specifications
# =============================================================================
master = Tk()
Label(master, text="Image File Name").grid(row=0)
Label(master,text="Metadata File Name").grid(row=1)
Label(master,text="Output File Name").grid(row=2)

fileName= Entry(master)
fileName.insert(10,'trainImage2.NTF')
fileName.grid(row=0, column=1)

metadata = Entry(master)
metadata.insert(10,'/home/grl/Documents/Trevor/ImageClassification/TrainingTool/atmCorrection/trainImage2.NTF')
metadata.grid(row=1,column=1)

outFileName = Entry(master)
outFileName.insert(10,'correctedImage.tif')
outFileName.grid(row=2,column=1)

Button(master, text='Continue', command=grabFileNames).grid(row=5, column=0, sticky=W, pady=4)
mainloop()

# =============================================================================
# Load model and raster files
# =============================================================================
mlp = pickle.load(open('atmCorrectionModel.pkl','rb'))

imageToClassify = gdal.Open(fileName)
metadataInfo = geoio.DGImage(metadata)
metadata = metadataInfo.meta
metadataFull = metadataInfo.meta_dg

gainOffsetData = np.array([[0.94,-5.809],[0.938,-4.996],[0.964,-3.021],[0.961,-5.522]])
sunInfo = metadataFull.IMD.IMAGE
sunVec = np.array([sunInfo.MEANSUNAZ, sunInfo.MEANSUNEL, sunInfo.MEANSATAZ, sunInfo.MEANSATEL, sunInfo.MEANOFFNADIRVIEWANGLE])

# =============================================================================
# Load, process, and predict data
# =============================================================================
print datetime.now()
colPerIt = 1
inFeatNum = imageToClassify.RasterCount
outFeatNum = 4
imageXSize = imageToClassify.RasterXSize
imageYSize = imageToClassify.RasterYSize
imageSize = imageXSize*imageYSize

sunBlockMat = np.tile(sunVec,(colPerIt*imageYSize,1))
pred = np.zeros((outFeatNum,imageYSize,imageXSize),dtype='float32')
hot = np.zeros((imageYSize*colPerIt,1))
tmpBlockHot = np.zeros((colPerIt*imageYSize,inFeatNum+1))
tmpBlockFull = np.zeros((colPerIt*imageYSize,inFeatNum+6))

for i in np.arange(imageXSize/colPerIt):
    tmpBlock = imageToClassify.ReadAsArray(i*colPerIt,0,colPerIt,imageYSize).astype('float32')
    tmpBlock = tmpBlock.reshape(inFeatNum,colPerIt*imageYSize).transpose()
    for j in range(inFeatNum):
        scalar = gainOffsetData[j,0]*metadata.abscalfactor[j]/metadata.effbandwidth[j]/10
        tmpBlock[:,j] = tmpBlock[:,j]*scalar+gainOffsetData[j,1]
    for j in range(imageYSize*colPerIt):
        vec = tmpBlock[j,:]
        m = vec[0]/vec[3]
        theta = np.arctan(m)
        hot[j] = vec[0]*np.sin(theta) - vec[1]*np.cos(theta)

    tmpBlockHot = np.append(tmpBlock,hot,axis=1)
    tmpBlockFull = np.append(tmpBlockHot,sunBlockMat,axis=1)

    tmpPred = mlp.predict(tmpBlockFull).transpose().reshape(outFeatNum,imageYSize,colPerIt)
    pred[:,:,i*colPerIt:(i+1)*colPerIt] = tmpPred
    print 'Predicted column %d of %d' %(i+1,imageXSize) 


# =============================================================================
# Write to tif file
# =============================================================================
driver = gdal.GetDriverByName('GTiff')
outdata = driver.Create(outFileName,imageXSize,imageYSize,outFeatNum,gdal.GDT_Float32)
print datetime.now()
outdata.SetGeoTransform(imageToClassify.GetGeoTransform())
outdata.SetProjection(imageToClassify.GetProjection())
for i in range(outFeatNum):
    outdata.GetRasterBand(i+1).WriteArray(pred[i,:,:])
outdata = None
imageToClassify = None


