# FAH
# A 'JrsContext' objects holds all the important global objects in _one_ object.
# So stuff like the config vars, the global logger objects, etc.
# ~FAH

import os
import shutil
import sys
from pathlib import Path

import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import hydra
import logging
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

import torch

# For visualizing models
import torchview

import clip

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset

import JrsUtilDataset
import JrsUtilModel

class Context:

    ###
    ### ( ) Constructor
    ###


    def __init__(self, cfg, originalWorkDir):

        #
        # ( ) Some 'general' stuff - config keys, logger etc.
        #

        # The config with all the config parameters
        self.cfg = cfg
        # Our 'global' hydra logger which we will use everywhere
        # Do we need another separate logger ?
        self.logger = logging.getLogger("global")

        # The original work dir (where 'main.py' resides), _before_ hydra changed the work directory ...
        # Note I explicitly enforce this behaviour also for Hydra versions >= 1.2, see the 'hydra' section of the config file ...
        self.originalWorkDir = originalWorkDir
        self.logger.info(f'Original work dir (= manifoldmixms root dir): {self.originalWorkDir}')
        # Current working dir (= 'run dir')
        self.runDir = os.getcwd()
        self.logger.info(f'Current work dir (= run dir): {self.runDir}')

        # Copy employed config file to runtime dir (current work dir)
        # See https://stackoverflow.com/questions/70064901/hydra-access-name-of-config-file-from-code
        hydraCfg = HydraConfig.get()
        cfgFileName = hydraCfg.job.config_name + '.yaml'
        shutil.copy(self.originalWorkDir + '/' + cfgFileName, self.runDir)


        #
        # ( )
        #

        # Number of individual models taken into account (will be set later)
        self.numberModels  = -1

        # Result files for the individual finetuned models
        self.individualModelsResultFile = 'individual_model_results.jsonl'

        # The result output file which we will write.
        self.manifoldMixResultFile = 'manifold_mix_results.json'

        #
        # ( ) Copy some needed files from original work dir to current dir
        #

        # Needed by 'ImageNet2p' class
        shutil.copy(self.originalWorkDir + '/imagenet_98_idxs.npy', self.runDir)

        # Precomputed results (accuracy on the datasets) for the ingredient models
        # Note if file does NOT exist in run dir, we copy it from root dir
        individualModelsResultFileFullPathRunDir = f'{self.runDir}/{self.individualModelsResultFile}'
        if not Path(individualModelsResultFileFullPathRunDir).is_file():
            shutil.copy(self.originalWorkDir + '/' + self.individualModelsResultFile, self.runDir)

        #
        # ( ) Initialize base model and the datasets used
        #

        self.initBaseModelAndDatasets()

        #
        # ( ) Sort the individual model (ingredients) in decreasing order (metric is accuracy on validation set)
        #

        self.logger.info("Sorting ingredient models by valAcc in decreasing order ...")
        self.sortedIngredientModels, self.sortedIngredientModelsValAcc = self.sortIngredientModelsByAccuracy()
        self.logger.info(f"Sorted ingredients: {self.sortedIngredientModels}")
        sortedIngredientModelsValAccStr = [f'{val:.6f}' for val in self.sortedIngredientModelsValAcc]
        self.logger.info(f"Ingredients valacc: {sortedIngredientModelsValAccStr}")

    ###
    ### ( ) Methods
    ###

    def logErrorAndExit(self, msg):
        self.logger.error(msg)
        exit(1)

    def initBaseModelAndDatasets(self):

        #
        # ( ) The 'base model' which defines the model architecture to be used, and its components
        #

        # Note 'self.preprocessModel' is neeeded as argument in the constructor of the dataset object
        self.logger.info(f'Loading base model: {self.cfg.model.base_model}')
        self.baseModel, self.preprocessModel = clip.load(self.cfg.model.base_model, 'cpu', jit=False)
        # JrsUtilModel.printModelStateDictKeys(self.baseModel)

        #
        # ( ) Create a dictionary with all needed dataset objects
        #

        self.datasetIdList = [ 'ImageNet2p', 'ImageNet', 'ImageNetV2', 'ImageNetSketch', 'ImageNetR', 'ObjectNet', 'ImageNetA' ]
        #self.datasetIdList = ['ImageNet2p']

        self.datasetObjMap = dict()
        for datasetId in self.datasetIdList:
            self.logger.info(f'Creating dataset: {datasetId}')
            self.datasetObjMap[datasetId] = JrsUtilDataset.createDataset(datasetId, self.preprocessModel,
                                self.originalWorkDir + "/" + self.cfg.datasets.rootdir,
                                self.cfg.valacc.batch_size, self.cfg.valacc.num_workers)

        # The dataset (typically the 'val split') used for calculating the validation accuracy of a specific model.
        self.valAccDatasetObj = self.datasetObjMap[self.cfg.valacc.dataset]
        self.logger.info(f'Validation dataset: {self.cfg.valacc.dataset}')

    # reads the ingreadient models and sort them by accuracy (in descending order)s
    def sortIngredientModelsByAccuracy(self):
        # Sort models by decreasing accuracy on the validation dataset (which is 'ImageNet2p' per default)
        individual_model_db = pd.read_json(self.individualModelsResultFile, lines=True)
        individual_model_val_accs = {}
        for _, row in individual_model_db.iterrows():
            individual_model_val_accs[row['model_name']] = row[self.cfg.valacc.dataset]
        individual_model_val_accs = sorted(individual_model_val_accs.items(), key=operator.itemgetter(1))
        individual_model_val_accs.reverse()
        sorted_models = [x[0] for x in individual_model_val_accs]
        individual_model_val_accs_values = [x[1] for x in individual_model_val_accs]
        self.numberModels = len(sorted_models)
        return sorted_models, individual_model_val_accs_values

    def loadStateDictForSortedModelWithIndex(self, idx):
        # FAH: It seems that for the individual ingredient models their state_dict (not the full model) has been saved.
        # See https://stackoverflow.com/questions/42703500/how-do-i-save-a-trained-model-in-pytorch
        return torch.load(os.path.join(f'{self.originalWorkDir}/{self.cfg.model.ingredients_root}',
                                       f'{self.sortedIngredientModels[idx]}.pt'))

    def getModelFromStateDict(self, stateDict):
        model = get_model_from_sd(stateDict, self.baseModel)
        return model

    # Calculates the (average) accuracy of the model 'model' on dataset with id 'datasetid'
    def calcModelAccuracy(self, model, datasetId):
        # Note regarding runtime: One 'valacc' call for the CLIP VIT-32-B model on the held out Imagenet validation dataset
        # ('ImageNet2p' in the code) takes roughly 30-40 seconds (on my ML PC with 2x Quadro A6000)
        accuracy = test_model_on_dataset(model, self.datasetObjMap[datasetId])
        return accuracy

    # Evaluates the model 'model' on all datasets.
    def evalModelOnDatasetsAndPrint(self, model, modelId, calculateAvgAccuracyOutOfDistribution = True):
        # list of accuracy values for all datasets except original ImageNet (so 'ImageNet2p' and 'ImageNet')
        # From this, we calculate the average 'out-of-distribution' accuracy, as in the Model soup paper
        accuracyAllExceptOutOfDistribution = [ ]
        accuracyForDatasetDict = dict()
        self.logger.info(f"Calculating accuracy of model '{modelId}' on available datasets ...")
        for datasetId in self.datasetIdList:
            # We do not eval on the 'valacc' dataset (usually 'ImageNet2p') which was used during both phases for the optimization
            if datasetId != self.cfg.valacc.dataset:
                accuracy = self.calcModelAccuracy(model, datasetId)
                self.logger.info(f"Accuracy of '{modelId}' on '{datasetId}': {accuracy:.6f}")
                if calculateAvgAccuracyOutOfDistribution:
                    if datasetId != 'ImageNet2p' and datasetId != 'ImageNet':
                        # We skip also ImageNet of course because this is not an out-of-distribution dataset
                        accuracyAllExceptOutOfDistribution.append(accuracy)
                accuracyForDatasetDict[datasetId] = accuracy
        if calculateAvgAccuracyOutOfDistribution:
            avgAccuracyAllExceptOutOfDistribution = sum(accuracyAllExceptOutOfDistribution) / len(accuracyAllExceptOutOfDistribution)
            self.logger.info(f"Average out-of-distribution accuracy: {avgAccuracyAllExceptOutOfDistribution:.6f}")
            accuracyForDatasetDict['average_out_of_dist'] = avgAccuracyAllExceptOutOfDistribution
        # We return also the accuracy on the test datasets and the average out-of-distribution accuracy as a dictionary
        return accuracyForDatasetDict

    # Visualize on ingredient model with 'torchview'.
    # We simple take the best ingredient model (with the highest validation accuracy)
    def visualizeOneIngredientModel(self):
        ingredientModel0StateDict = self.loadStateDictForSortedModelWithIndex(0)
        ingredientModel0 = self.getModelFromStateDict(ingredientModel0StateDict)
        # Set a breakpoint in fn. 'test_model_on_dataset' from 'utils.py' in order to see thee acdtual size of the input
        #valAcc = self.calcModelAccuracy(ingredientModel0, self.cfg.valacc.dataset)
        # Use 'torchview' package to visualize the model.
        # See https://github.com/mert-kurttutan/torchview
        #batch_size = self.cfg.valacc.batch_size
        batch_size = 256
        # device='meta' -> no memory is consumed for visualization
        #print(ingredientModel0)
        # The CLIP variant we are using has 224x224 pixel images as input
        model_graph = torchview.draw_graph(ingredientModel0, input_size=(batch_size, 3, 224, 224), depth = 3)
        # set graph output format to 'png' instead of default PDF
        #model_graph.visual_graph.format = 'png'
        # See https://graphviz.readthedocs.io/en/stable/manual.html
        model_graph.visual_graph.render(filename='model_visualization')
        xyz = 123

