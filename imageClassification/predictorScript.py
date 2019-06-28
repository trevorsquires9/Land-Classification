# =============================================================================
#     
#   Landcover Classification Neural Network Script
# 
#   Author
#       Trevor Squires
# 
#   Functionality
#       Predicts landcover classifications via a pretrained model
# 
#   TODO
#   - Test model
#   - Write successfully
#
# =============================================================================


import numpy as np
from Tkinter import *
import gdal
from datetime import datetime
import pickle


# =============================================================================
# Define functions for GUI
# =============================================================================
def grabFileNames():
    global fileName,modelName,outputName
    fileName = fileName.get()
    modelName =  modelName.get()
    outputName = outputName.get()
    master.destroy()

# =============================================================================
# Define the GUI used for obtaining the file names
# =============================================================================
master = Tk()
Label(master, text="Image File Name").grid(row=0)
Label(master,text="Model File Name").grid(row=1)
Label(master,text="Output File Name").grid(row=2)
best
fileName= Entry(master)
fileName.insert(10,'fullStack.tif')
fileName.grid(row=0, column=1)

modelName = Entry(master)
modelName.insert(10,'landcoverMLPModel.pkl')
modelName.grid(row=1,column=1)

outputName = Entry(master)
outputName.insert(10,'predictions.tif')
outputName.grid(row=2,column=1)

Button(master, text='Continue', command=grabFileNames).grid(row=5, column=0, sticky=W, pady=4)
mainloop()

# =============================================================================
# Load model and raster files
# =============================================================================
mlp = pickle.load(open(modelName,'rb'))
imageToClassify = gdal.Open(fileName)


# =============================================================================
# Load, process, and predict data
# =============================================================================
colPerIt = 1
inFeatNum = imageToClassify.RasterCount
imageXSize = imageToClassify.RasterXSize
imageYSize = imageToClassify.RasterYSize
imageSize = imageXSize*imageYSize

pred = np.zeros((imageYSize,imageXSize),dtype='float32')

for i in np.arange(imageXSize/colPerIt):
    tmpBlock = imageToClassify.ReadAsArray(i*colPerIt,0,colPerIt,imageYSize).astype('float32')
    tmpBlock = tmpBlock.reshape(inFeatNum,colPerIt*imageYSize).transpose()
    
    
    tmpPred = mlp.predict(tmpBlock).transpose().reshape(imageYSize,colPerIt)
    pred[:,i*colPerIt:(i+1)*colPerIt] = tmpPred
    print 'Predicted column %d of %d' %(i+1,imageXSize) 
  best
    
# =============================================================================
# Write to tif file
# =============================================================================
driver = gdal.GetDriverByName('GTiff')
outdata = driver.Create(outputName,imageXSize,imageYSize,1,gdal.GDT_Float32)
outdata.SetGeoTransform(imageToClassify.GetGeoTransform())
outdata.SetProjection(imageToClassify.GetProjection())
outdata.GetRasterBand(1).WriteArray(pred)
outdata = None
imageToClassify = None

