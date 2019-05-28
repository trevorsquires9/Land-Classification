import numpy as np
import sklearn
#from sklearn.ensemble import RandomForestClassifier
import gdal
from gdalconst import GDT_Byte

#######################Read image and predict classes
def rf_class(b,ob,m,row,col,feats): #Read in image and predict class
	xBlockSize = col
	yBlockSize = 32 # you might alter this for optimization, if the code is slow.
	for i in range(0, row, yBlockSize):
		if i + yBlockSize < row:
			numRows=yBlockSize
		else:
			numRows=row-i
		for j in range(0, col, xBlockSize):
			if j + xBlockSize < col:
				numCols = xBlockSize
			else:
				numCols = col-j
			print "Loading arrays"
			dat=np.empty((numCols*numRows,feats),dtype='f4')
			for k in range(0,feats):
				tmp=b[k].ReadAsArray(j,i,numCols,numRows)
				tmp=tmp.reshape(numCols*numRows,)
				dat[:,k]=tmp
			print "predicting"
			filt=m.predict(dat)
			filt=filt.astype('u1')
			filt=filt.reshape(numRows,numCols)
			#filt[idx[0],idx[1]]=65535
			print "Writing %s %s" %(i+numRows,j+numCols)
			ob.WriteArray(filt,j,i)
			ob.FlushCache()
   
############################################Load training, define params, train classififer

#training_file = '/home/klasko/Documents/randomforest/training/subset_training_RF3.txt'
training_file = '/home/klasko/Documents/randomforest/20mdata/subset_training_RF3_VV_and_VH_20m.txt' #Should be text file with first column with class code i.e. 1, 2, 3. Remaining columns are the corresponding remote sensing values for that class. Make sure that the column order in this file matches the order for the composite raster stack (excluding the first column w/ class code).
dat = np.loadtxt(training_file, dtype = 'float', delimiter = ',', skiprows = 1, usecols = (1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43))

# You can consider to replace the RF classifier with a CART/Decision tree (see http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html)
rf = sklearn.ensemble.RandomForestClassifier(n_estimators=500, criterion='gini', max_features='auto', max_leaf_nodes=None, min_impurity_split=1e-03, bootstrap=True, class_weight='balanced', n_jobs=-1) #n_jobs is for multiprocessing. Set to -1 uses all available cores.
rf.fit(dat[:,1:dat.shape[1]],dat[:,0])
#You may need to customize the randomForestClassifier or allow the different inputs to be variables depending on the application. i.e. n_estimators, etc. 


###########################################Predict classes and write out raster

driver = gdal.GetDriverByName('GTiff')
driver.Register()
names=np.genfromtxt('/home/klasko/Documents/randomforest/bandlist_VV_VH.txt',dtype='S') #Uses a text file of the band names i.e. band1, band2
feats = 42 #number of bands in the input image to be classified


ras = '/home/klasko/Documents/randomforest/20mdata/S1A_MERGED_VV_VH_20m.tif' #Replace with path to your composite/stack of input image to be used for classification
ds=gdal.Open(ras)
bands=list()
featss = feats + 1
for i in range(1,featss): #43 is the number of bands in my raster stack + 1
	bands.append(ds.GetRasterBand(i))	
c=ds.RasterXSize
r=ds.RasterYSize
geotransform=ds.GetGeoTransform()
proj=ds.GetProjection()
out=driver.Create('/home/klasko/Documents/randomforest/aatest.tif', c,r,1,GDT_Byte, ['COMPRESS=LZW']) #Replace path and filename with your own choice, #Make sure that this is the correct way to add a compression in gdal.
out.SetProjection(proj)
out.SetGeoTransform(geotransform)
outband=out.GetRasterBand(1)
outband.SetNoDataValue(65535)

#Main
rf_class(bands,outband,rf,r,c,feats)
out=None

del out