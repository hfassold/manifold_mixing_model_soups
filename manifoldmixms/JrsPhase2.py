# A Context object capturing the stuff for 'phase 2'

import copy
import os
import sys
import shutil
import pathlib
from pathlib import Path

import numpy as np
import json


import nevergrad as ng

import JrsUtilDataset
import JrsUtilMisc
import JrsUtilModel
import JrsUtilOptim

import JrsPhase1

class Context:
    def __init__(self, ctxMaster, ctxPhase1):
        self.ctx = ctxMaster
        self.ctx.logger.info('---------------------------')
        self.ctx.logger.info('Starting phase 2 ...')

        self.ctxPhase1 = ctxPhase1

        # Make a deep copy of some of the results of phase 1 (see there for documentation what they are)
        self.ingredientIndices = copy.deepcopy(self.ctxPhase1.ingredientIndices)
        self.indicesNearMisses = copy.deepcopy(self.ctxPhase1.indicesNearMisses)
        self.beta = np.copy(self.ctxPhase1.beta)

        # A list with the state dicts of the loaded models (corresponding to the indices defined in 'self.ingredientIndices')
        self.ingredientModelsStateDicts = [ ]

        # ( ) The 'myu' 2D-array (numpy array) calculated in phase 2.
        self.myu = None

    def run(self):

        #
        # ( ) Some preparations
        #

        self.ctx.logger.info('Loading all ingredient models...')
        # Note all ingredient models are loaded on CPU
        for ingredientIdx in self.ingredientIndices:
            ingredientModelStateDict = self.ctx.loadStateDictForSortedModelWithIndex(ingredientIdx)
            self.ingredientModelsStateDicts.append(ingredientModelStateDict)

        # Now we can determine the components. Note we cannot use the 'base (CLIP) model',
        # as I noticed that it misses some components (e.g. final classification head)
        self.components, self.componentsIndices = JrsUtilModel.getComponentsFromComponentGranularity(self.ingredientModelsStateDicts[0],
                                                        self.ctx.cfg.model.base_model, self.ctx.cfg.phase2.component_granularity)
        self.numberComponents = len(self.components)

        #
        # ( ) Now do the optimization over 'myu'
        #

        # The start value (guess) for factors myu to be optimized is simply 1.0
        myuStart = np.ones(shape = self.beta.shape)
        myuOpt, accuracyOpt = self.doOptimization(myuStart)
        self.myu = myuOpt

        #
        # ( ) Final steps
        #

        self.fusedModelStateDict = self.getFusedModelStateDictFromMyu(self.myu)
        # ( ) Evaluate accuracy of fused model from phase 2 on our datasets
        fusedModelPhase2 = self.ctx.getModelFromStateDict(self.fusedModelStateDict)
        self.ctx.evalModelOnDatasetsAndPrint(fusedModelPhase2, 'fusedModelPhase2')
        # In order to save memory, we reset the list containing the state dicts of the ingredient models for phase 2s
        self.ingredientModelsStateDicts = []
        # Serialize all phase 2 results which shall be kept up to JSON
        self.saveResultsToJson()

    # Used inside optimizer, returns the 'fn. value' for current parameter set 'myu'
    # Note as Nevergrad provides only minimization, we return '1 - accuracy' instead of accuracy
    def evalFunctionWithMyu(self, myu):
        F = self.getFusedModelStateDictFromMyu(myu)
        modelForF = self.ctx.getModelFromStateDict(F)
        accuracy = self.ctx.calcModelAccuracy(modelForF, self.ctx.cfg.valacc.dataset)
        fnVal = 1.0 - accuracy
        return fnVal

    def doOptimization(self, myuStart):
        #self.ctx.logger.info('Starting optimization ...')
        # ( ) Calculate 'budget' in terms of maximum allowed function evaluation
        maxFunctionCallsTotal = -1
        if self.ctx.cfg.phase2.optimizer.max_func_evals_per_dim == 0:
            # Special value signals that we should do only one function evaluations in total
            maxFunctionCallsTotal = 1
        else:
            numberIngredients = len(self.ingredientIndices)
            maxFunctionCallsTotal = numberIngredients * self.numberComponents * self.ctx.cfg.phase2.optimizer.max_func_evals_per_dim

        myuBounds = [ self.ctx.cfg.phase2.optimizer.bounds[0], self.ctx.cfg.phase2.optimizer.bounds[1] ]
        ngParamMyu = ng.p.Array(init = myuStart, lower = myuBounds[0], upper = myuBounds[1])
        # The instrumentation defines the parameters over which shall be optimized and other stuff.
        instrumentation = ng.p.Instrumentation(myu = ngParamMyu)
        instrumentation.random_state.seed(self.ctx.cfg.phase2.optimizer.random_seed)
        # See https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimizers.base.Optimizer
        # Note the 'NGOpt' optimizer is the one which is _automatically_ chosen (by a meta-heuristic) in an optimal way
        # for the problem at hand (depends on the type of the variables, the function budget etc.)
        self.optimizer = ng.optimizers.NGOpt(parametrization = instrumentation, budget = maxFunctionCallsTotal)
        self.ctx.logger.info(f"Starting {self.optimizer.optim.name} optimizer, fn_evals = {maxFunctionCallsTotal}")
        # Now run optimizer.
        # Note it seems that the function give to optimizer may NOT have additional parameters (except for 'self')
        # If you need more parameters, you have to use a workaround (see 'doOptimization' function in Phase 1).
        ngParamMyuOpt = self.optimizer.minimize(self.evalFunctionWithMyu, verbosity = self.ctx.cfg.phase2.optimizer.verbosity)
        # See https://facebookresearch.github.io/nevergrad/optimization.html
        # Note I think the first component of tuple 'ngParamLambtaOpt.value' (which I ignore)
        # is because the function 'evalFunctionWithLambda' is a class member function (with implicit parameter 'self')
        dummy, paramOpt = ngParamMyuOpt.value
        myuOpt = paramOpt['myu'].astype(np.float32)
        accuracyOpt = 1.0 - ngParamMyuOpt.loss
        self.ctx.logger.info(f'ValAcc(FusedModel)): {accuracyOpt:.6f}')
        return myuOpt, accuracyOpt

    # We define the fused model F now via the component-wise _convex_ combination of the K models
    # F = sum (C_1 * myu_1 * beta_1 * M_1  + C_2 * myu_2 * beta_2 * M_2 + … + C_K * myu_K * beta_K* M_K).
    # The normalization factors C_j = (myu_j * b_j) / P, with P = sum_over_all_j{myu_j * beta_j}, are needed
    # to ensure that  for each component we have a convex combination where all weights sum up to 1.
    # The ‘base vectors’ beta_j have been calculated in phase 1.
    def getFusedModelStateDictFromMyu(self, myuMat):
        P = np.ones(self.numberComponents)
        if self.ctx.cfg.phase2.enforce_convex_combination:
            numberIngredients = len(self.ingredientIndices)
            for i in range(self.numberComponents):
                weightSum = 0.0
                for j in range(numberIngredients):
                    weightSum += myuMat[j, i] * self.beta[j, i]
                P[i] = weightSum
        fusedModelStateDict = dict()
        for currIngredientCounter in range(numberIngredients):
            # Load current model state-dictto be added
            currModelStateDict = self.ingredientModelsStateDicts[currIngredientCounter]
            for currLayer in currModelStateDict:
                # Get the component (and its index) to which the current layer belongs to
                currLayerComponent, currLayerComponentIdx = JrsUtilModel.getComponentInfoForLayer(currLayer,
                                                                self.components, self.componentsIndices)
                factor = (1.0 / P[currLayerComponentIdx]) * myuMat[currIngredientCounter, currLayerComponentIdx] *  self.beta[currIngredientCounter, currLayerComponentIdx]
                if currIngredientCounter == 0:
                    # We are at the start
                    fusedModelStateDict[currLayer] = currModelStateDict[currLayer].cpu().clone() * factor
                else:
                    fusedModelStateDict[currLayer] = fusedModelStateDict[currLayer].cpu().clone() + currModelStateDict[currLayer].cpu().clone() * factor
        return fusedModelStateDict

    # Saves (serializes) the results of phase 2 into a JSON file, in run dir (usually)
    def saveResultsToJson(self):
        self.ctx.logger.info(f"Saving results of phase 2 to file: {self.ctx.cfg.phase2.result_filename} ")
        resultDict = {
            'numberComponents': self.numberComponents,
            'ingredientIndices': self.ingredientIndices,
            'indicesNearMisses': self.indicesNearMisses,
            'beta': JrsUtilMisc.getDictFromNumpy2DArray(self.beta),
            'myu': JrsUtilMisc.getDictFromNumpy2DArray(self.myu),
            # Note 'self.fusedModelStateDict' is not saved, as we can re-construct it from 'myu', 'beta' and 'ingedientIndices'
        }
        # See https://stackoverflow.com/questions/17043860/how-to-dump-a-dict-to-a-json-file
        with open(self.ctx.cfg.phase2.result_filename + '.json', 'w') as jsonFile:
            json.dump(resultDict, jsonFile, indent=2)




