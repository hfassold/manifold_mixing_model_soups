# packages in the local directory
from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ObjectNet, ImageNetA

def createDataset(datasetId, preprocess, dataLocation, batchSize, numWorkers):
    mapStrToClass = { 'ImageNet2p': ImageNet2p, 'ImageNet' : ImageNet, 'ImageNetV2' : ImageNetV2,
                      'ImageNetSketch' : ImageNetSketch, 'ImageNetR' : ImageNetR, 'ObjectNet' : ObjectNet, 'ImageNetA' : ImageNetA }
    datasetClass = mapStrToClass[datasetId]
    datasetObj = datasetClass(preprocess, dataLocation, batchSize, numWorkers)
    return datasetObj