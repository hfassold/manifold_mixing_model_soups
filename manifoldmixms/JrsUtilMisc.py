# Diverse utility functions used throughout the program

import numpy as np

def getNumpy2DArrayFromDict(myDict):
    # See https://stackoverflow.com/questions/38101292/python-dict-to-numpy-multidimensional-array
    numberRows = len(myDict)
    numberCols = len(myDict['row_0'])
    npArray = np.zeros(shape = (numberRows, numberCols), dtype = np.float32)
    for i in range(numberRows):
        npArray[i, :] = myDict['row_' + str(i)]
    return npArray

def getDictFromNumpy2DArray(myArray):
    myDict = dict()
    numberRows, numberCols = myArray.shape
    for i in range(numberRows):
        myDict['row_' + str(i)] = myArray[i, :].tolist()
        i = i + 1
    return myDict

