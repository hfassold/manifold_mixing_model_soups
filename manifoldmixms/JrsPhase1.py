# A Context object capturing the stuff for 'phase 1'
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


class Context:
    def __init__(self, ctxMaster):

        # ( ) The master context, containing the config etc.
        self.ctx = ctxMaster
        self.ctx.logger.info('---------------------------')
        self.ctx.logger.info('Starting phase 1 ...')

        # ( ) Get the 'components' of the model architecure (and their indices).
        sampleIngredientModelStateDict = self.ctx.loadStateDictForSortedModelWithIndex(0)
        # Now we can determine the components. Note we cannot use the 'base (CLIP) model',
        # as I noticed that it misses some components (e.g. final classification head)
        self.components, self.componentsIndices = JrsUtilModel.getComponentsFromComponentGranularity(sampleIngredientModelStateDict,
                                                                    self.ctx.cfg.model.base_model, self.ctx.cfg.phase1.component_granularity)
        self.numberComponents = len(self.components)

        # ( ) The indices of the actual ingredients used in phase 1.
        # They refer to the the model list (sorted by Valacc) stored in self.ctx.sortedIngredientModels
        self.ingredientIndices = [ ]

        # ( ) These are the indices for which the optimizer was invoked _without_ success.
        # So these models are the 'near misses', seem to be also good
        # We might use this in phase 3 ...
        self.indicesNearMisses = []

        # ( ) This is the (exact) average of the 'near-misses' ingredient models in 'self.indicesNearMisses'
        self.avgModelNearMissesStateDict = None

        # ( ) The 'beta' 2D-array (numpy array) calculated in phase 1.
        # The beta values are used as start values for the optimizer in phase 2.
        self.beta = None

        # ( ) The state dict of the final 'fused' model (= the final accumulator model9
        self.fusedModelStateDict = None

    def run(self):

        #
        # ( ) Some preparations
        #

        self.ctx.logger.info('Checking the ingredient models ...')
        # For the adaptive subsamping in the indices. 'currentSegmentIdx' is the index within current segment.
        subsamplingFactor = 1
        currentSegmentIdx = 0

        numberModelsToInspect = self.ctx.cfg.phase1.number_models_to_inspect
        if numberModelsToInspect > self.ctx.numberModels:
            numberModelsToInspect = self.ctx.numberModels

        # 'idxTakeAlways' Is the index of the ingredient which we take 'always'.
        # Usually, this is the index 0 (corresponds to model with the highest valacc).
        idxTakeAlways = 0
        candidateMode = self.ctx.cfg.phase1.candidate_mode
        candidateIndicesToInspect = list(range(1, numberModelsToInspect))
        if candidateMode == 0:
            # Nothing to do, keep list as it is
            xyz = 123
        elif candidateMode == 1:
            candidateIndicesToInspect.reverse()
        elif candidateMode == 5:
            # Make a random permutation of the indices.
            # See https://stackoverflow.com/questions/47742622/np-random-permutation-with-seed
            candidateIndicesToInspect = np.random.RandomState(seed = ctx.cfg.cfg.phase1.random_seed).permutation(numberModelsToInspect)
            idxTakeAlways = candidateIndicesToInspect[0]
            candidateIndicesToInspect.pop(0)
        else:
            raise ValueError("NOT IMPLEMENTED YET")

        # Note we _always_ pick the the model with index 'idxTakeAlways' (usually, this is id. 0, so the model with best valacc)
        self.ingredientIndices.append(idxTakeAlways)
        # The (state dict) of the current 'accumulator' model A
        A = self.ctx.loadStateDictForSortedModelWithIndex(idxTakeAlways)
        # The accuracy (valacc) for current 'accumulator' model A
        accuracyA = self.ctx.calcModelAccuracy(self.ctx.getModelFromStateDict(A),  self.ctx.cfg.valacc.dataset)
        self.ctx.logger.info(f'Adding ingredient with index {idxTakeAlways} and ValAcc {accuracyA:.6f}')
        # 'lambdaAll' matrix is iteratively extended (row per row) each time when a component is added
        # It keeps up the 'optimal' lambda coefficients calculated from the optimizer
        lambdaAll = None

        #
        # ( ) Try out the candidate models C in a sequential way
        #

        self.ctx.logger.info(f'Number of (best-performing) models to inspect: {numberModelsToInspect}')

        for idx in candidateIndicesToInspect:
            if idx % subsamplingFactor == 0:
                self.ctx.logger.info(f'ValAcc(A): {accuracyA:.6f}')
                # Try out candidate 'C'
                accuracyC = self.ctx.sortedIngredientModelsValAcc[idx]
                self.ctx.logger.info (f'Investigating whether candidate C with index {idx} and ValAcc {accuracyC:.6f} should be added ...')
                C = self.ctx.loadStateDictForSortedModelWithIndex(idx)
                # Short-circuit: If 'valacc(avg(A,C)) < tau * valacc(A)', then _discard_ candidate C.
                # Because then it's not likely that we get an improvement from the optimization step
                approxAvgModel = self.ctx.getModelFromStateDict(self.getStateDictForApproxAvgModel(A, C))
                accuracyApproxAvgModel = self.ctx.calcModelAccuracy(approxAvgModel, self.ctx.cfg.valacc.dataset)
                self.ctx.logger.info(f'ValAcc(approxAvg(A, C)): {accuracyApproxAvgModel:.6f}')
                if accuracyApproxAvgModel >= self.ctx.cfg.phase1.tau * accuracyA:
                    # We do the optimization step
                    K = len(self.ingredientIndices)
                    # The start value (guess) for the factors to be optimized is simpliy 'K / K + 1)
                    # This correponds (see engineer notes) roughly to the average of all ingredients (inclusive candidate 'C'),
                    # if we assume that the optimizer did not change the start values dramatically.
                    lambdaStart = np.ones(shape=(1, self.numberComponents)) * (K / (K + 1))
                    # Now do the optimization, in order to calculate the optimal lambda factors
                    lambdaOpt, accuracyOpt = self.doOptimization(A, C, lambdaStart)
                    #self.ctx.logger.info(f"ValAccOpt after optimization: {accuracyOpt}")
                    if accuracyOpt >= accuracyA:
                        if self.ctx.cfg.phase1.do_potential_second_optimizer_pass:
                            accuracyOptBefore = accuracyOpt
                            # We were successfull, do a second optimizer pass to get (hopefully) an even better result
                            self.ctx.logger.info(f'Doing second optimizer pass')
                            lambdaOpt, accuracyOpt = self.doOptimization(A, C, lambdaOpt, isFirstPass = False)
                            if accuracyOpt >= (1.0 + 0.3 * (1.0 - self.ctx.cfg.phase1.tau)) * accuracyOptBefore:
                                # There is potential benefit also of a third pass. So do it.
                                self.ctx.logger.info(f'Doing third optimizer pass')
                                lambdaOpt, accuracyOpt = self.doOptimization(A, C, lambdaOpt, isFirstPass = False)
                        self.ctx.logger.info(f"Adding ingredient model with index {idx} to soup")
                        # New accumulated model is _better_ than the current one.
                        # So we add candidate 'C' to the ingredients !
                        self.ingredientIndices.append(idx)
                        # Calculate new accumulated model via:
                        # A' = lambdaOpt * A + (1 - lambdaOpt) * C
                        A = self.getFusedModelStateDictFromLambda(A, C, lambdaOpt)
                        accuracyA = accuracyOpt
                        # Append 'lambdaOpt' as last row of 'lambdaAll'
                        if lambdaAll is None:
                            lambdaAll = lambdaOpt
                        else:
                            lambdaAll = np.vstack([lambdaAll, lambdaOpt])
                        self.ctx.logger.info(f"Current ingredients: {self.ingredientIndices}")
                    else:
                        # Adding candidate C would not bring an imporvement to current model 'A', so we discard it
                        self.ctx.logger.info(f'Optimization not successfull, skipping Candidate C')
                        # Calculate an exact average model of the 'near misses' (experimental)
                        self.updateAvgModelNearMissesStateDict(C)
                        self.indicesNearMisses.append(idx)
                        avgModelNearMisses = self.ctx.getModelFromStateDict(self.avgModelNearMissesStateDict)
                        accuracyAvgModelNearMisses = self.ctx.calcModelAccuracy(avgModelNearMisses, self.ctx.cfg.valacc.dataset)
                        self.ctx.logger.info(f'ValAcc(avgModelNearMisses): {accuracyAvgModelNearMisses:.6f}')
                        approxAvgWithNearMissesModel = self.ctx.getModelFromStateDict(self.getStateDictForApproxAvgModel(A, self.avgModelNearMissesStateDict))
                        accuracyApproxAvgWithNearMissesModel = self.ctx.calcModelAccuracy(approxAvgWithNearMissesModel, self.ctx.cfg.valacc.dataset)
                        self.ctx.logger.info(f'ValAcc(approxAvg(A, avgModelNearMisses)): {accuracyApproxAvgWithNearMissesModel:.6f}')


                else:
                    # Accumulator model 'A' etc. are not changed
                    self.ctx.logger.info('Short-circuit: Skipping optimization step for candidate C')


            else:
                xyz = 123
                # We skip this model due to our adaptive subsampling (see config parameter 'cfg.phase1.segment_size')

            # adapt subsampling factors
            currentSegmentIdx += 1
            if currentSegmentIdx >= subsamplingFactor * self.ctx.cfg.phase1.segment_size:
                # It's time to increase the subsampling factor by 1
                subsamplingFactor += 1
                currentSegmentIdx = 0

        #
        # ( ) Final steps (convert lambdas to beta etc)
        #

        self.beta = self.convertLambdaToBeta(lambdaAll)
        self.fusedModelStateDict = self.getFusedModelStateDictFromBeta(self.ingredientIndices, self.beta)
        # We save also the final accumulated 'A' (state dict). It should be identical to 'self.fusedModelStateDict' !
        self.finalAccumA = A
        # ( ) Evaluate accuracy of fused model from phase 1 on our datasets
        fusedModelPhase1 = self.ctx.getModelFromStateDict(self.fusedModelStateDict)
        self.ctx.evalModelOnDatasetsAndPrint(fusedModelPhase1, 'fusedModelPhase1')
        # Serialize all phase 1 results which shall be kept up to JSON
        self.saveResultsToJson()

    # Used inside optimizer, returns the 'fn. value' for current parameter set 'lambda'
    # Note as Nevergrad provides only minimization, we return '1 - accuracy' instead of accuracy
    # Note unfortunately, it seems that nevergrad functions cannot have  'constant' parameters (like here 'A' and 'C')
    # Therefore, we save 'A' and 'C' in the class object itself temporarly, as 'self.tmp_A' and 'self.tmp_C'.
    def evalFunctionWithLambda(self, lambta):
        A1 = self.getFusedModelStateDictFromLambda(self.tmp_A, self.tmp_C, lambta)
        modelForA1 = self.ctx.getModelFromStateDict(A1)
        accuracy = self.ctx.calcModelAccuracy(modelForA1, self.ctx.cfg.valacc.dataset)
        fnVal = 1.0 - accuracy
        return fnVal

    def doOptimization(self, A, C, lambdaStart, isFirstPass = True):
        #self.ctx.logger.info('Starting optimization ...')
        # ( ) Calculate 'budget' in terms of maximum allowed function evaluation
        maxFunctionCallsTotal = -1
        if self.ctx.cfg.phase1.optimizer.max_func_evals_per_dim == 0:
            # Special value signals that we should do only one function evaluations in total
            maxFunctionCallsTotal = 1
        else:
            maxFunctionCallsTotal = self.numberComponents * self.ctx.cfg.phase1.optimizer.max_func_evals_per_dim
        # ( ) Calculate lower and upper bound for each 'lambda'. Note that lambdaStart value is 'K / K + 1' (for each value)
        if self.ctx.cfg.phase1.optimizer.bounds_delta < 0.0 or self.ctx.cfg.phase1.optimizer.bounds_delta > 1.0:
            self.ctx.logErrorAndExit("Config parameter phase1.bounds_delta must be in unit range [0, 1]")
        if isFirstPass:
            deltaForLambda = self.ctx.cfg.phase1.optimizer.bounds_delta * (1 - lambdaStart[0, 0])
            lambdaBounds = [lambdaStart[0, 0] - deltaForLambda, lambdaStart[0, 0] + deltaForLambda]
        else:
            # In second (or third) pass, we cannot (as in first pass) assume that all values in 'lambdaStart' have the same value.
            # Furthermore, I don't know whether nevergrad allows _different_ bound for every variable.
            # In order to keep it simple, we therefore use simply the interval [0.3 1.0] as bounds
            lambdaBounds = [0.3, 1.0]
            # For safety, we clip 'lambdaStart' to this interval
            lambdaStart = np.clip(lambdaStart, lambdaBounds[0], lambdaBounds[1])
        # Regading initialization of a parameter (or parameter array) with a start value, and setting bounds for it, see
        # https://stackoverflow.com/questions/67269671/optimize-function-that-takes-numpy-array-using-nevergrad
        # https://github.com/facebookresearch/nevergrad/issues/1066
        # https://github.com/facebookresearch/nevergrad/issues/482
        # https://facebookresearch.github.io/nevergrad/parametrization.html
        # See https://facebookresearch.github.io/nevergrad/parametrization_ref.html#nevergrad.p.Array
        ngParamLambta = ng.p.Array(init = lambdaStart, lower = lambdaBounds[0], upper = lambdaBounds[1])
        # The instrumentation defines the parameters over which shall be optimized and other stuff.
        instrumentation = ng.p.Instrumentation(lambta = ngParamLambta)
        instrumentation.random_state.seed(self.ctx.cfg.phase1.optimizer.random_seed)
        # See https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimizers.base.Optimizer
        # Note the 'NGOpt' optimizer is the one which is _automatically_ chosen (by a meta-heuristic) in an optimal way
        # for the problem at hand (depends on the type of the variables, the function budget etc.)
        self.optimizer = ng.optimizers.NGOpt(parametrization = instrumentation, budget = maxFunctionCallsTotal)
        self.ctx.logger.info(f"Starting {self.optimizer.optim.name} optimizer, fn_evals = {maxFunctionCallsTotal}")
        # Now run optimizer.
        # See https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.optimizers.base.Optimizer.minimize
        # Note unfortunately, it seems that nevergrad functions cannot have  'constant' parameters (like here 'A' and 'C')
        # Therefore, we save 'A' and 'C' in the class object itself temporarly, as 'self.tmp_A' and 'self.tmp_C'. This is a workaround of course.
        self.tmp_A = A
        self.tmp_C = C
        ngParamLambtaOpt = self.optimizer.minimize(self.evalFunctionWithLambda, verbosity = self.ctx.cfg.phase1.optimizer.verbosity)
        # See https://facebookresearch.github.io/nevergrad/optimization.html
        # Note I think the first component of tuple 'ngParamLambtaOpt.value' (which I ignore)
        # is because the function 'evalFunctionWithLambda' is a class member function (with implicit parameter 'self')
        dummy, paramOpt = ngParamLambtaOpt.value
        lamtaOpt = paramOpt['lambta'].astype(np.float32)
        accuracyOpt = 1.0 - ngParamLambtaOpt.loss
        self.ctx.logger.info(f'ValAcc(lambdaOpt * A + (1-lambdaOpt) * C)): {accuracyOpt:.6f}')
        return lamtaOpt, accuracyOpt

    def getStateDictForApproxAvgModel(self, A, C):
        # Note: Assumes that candidate 'C' has NOT been yet added to ingredient indices
        # returns for each layer: (K / (K + 1)) * A + (1 / (K + 1)) * C
        # This would correspond approximately to the average of the 'K + 1' components (so including 'C')
        # (_if_ the optimization step did not change the start values too much ...)
        # See my Enginer Notes document for more details.
        approxAvg = dict()
        K = len(self.ingredientIndices)
        factorA = (K / (K + 1.))
        for currLayer in A:
            approxAvg[currLayer] = A[currLayer].cpu().clone() * factorA + C[currLayer].cpu().clone() * (1.0 - factorA)
        return approxAvg


    # 'self.avgModelNearMissesStateDict' is the _exact_ average of all ingredeint models in 'indicesNearMisses' _plus_ C
    def updateAvgModelNearMissesStateDict(self, C):
        exactAvg = dict()
        K = len(self.indicesNearMisses)
        if K == 0:
            self.avgModelNearMissesStateDict = copy.deepcopy(C)
        else:
            factor = (K / (K + 1.))
            for currLayer in C:
                exactAvg[currLayer] = self.avgModelNearMissesStateDict [currLayer].cpu().clone() * factor + C[currLayer].cpu().clone() * (1.0 - factor)
            self.avgModelNearMissesStateDict = exactAvg

    def getStateDictForApproxAvgModel(self, A, C):
        # Note: Assumes that candidate 'C' has NOT been yet added to ingredient indices
        # returns for each layer: (K / (K + 1)) * A + (1 / (K + 1)) * C
        # This would correspond approximately to the average of the 'K + 1' components (so including 'C')
        # (_if_ the optimization step did not change the start values too much ...)
        # See my Enginer Notes document for more details.
        approxAvg = dict()
        K = len(self.ingredientIndices)
        factorA = (K / (K + 1.))
        for currLayer in A:
            approxAvg[currLayer] = A[currLayer].cpu().clone() * factorA + C[currLayer].cpu().clone() * (1.0 - factorA)
        return approxAvg

    # Calculate A' = lambda * A + (1 - lambda) * C and return it
    def getFusedModelStateDictFromLambda(self, A, C, lambta):
        newA = dict()
        for currLayer in A:
            # Get the component (and its index) to which the current layer belongs to
            currLayerComponent, currLayerComponentIdx = JrsUtilModel.getComponentInfoForLayer(currLayer,
                                                            self.components, self.componentsIndices )
            lambta_i = lambta[0, currLayerComponentIdx]
            newA[currLayer] = A[currLayer].cpu().clone() * lambta_i + C[currLayer].cpu().clone() * (1.0 - lambta_i)
        return newA

    # Returns beta_1 * m_1 + ... beta_n * m_n, where m_j are the ingredient models which were selected.
    def getFusedModelStateDictFromBeta(self, ingredientIndices, betaMat):
        fusedModelStateDict = dict()
        for currIngredientCounter in range(len(ingredientIndices)):
            # Load current model state-dictto be added 
            currModelStateDict = self.ctx.loadStateDictForSortedModelWithIndex(ingredientIndices[currIngredientCounter])
            for currLayer in currModelStateDict:
                # Get the component (and its index) to which the current layer belongs to
                currLayerComponent, currLayerComponentIdx = JrsUtilModel.getComponentInfoForLayer(currLayer,
                                                                self.components, self.componentsIndices )
                beta = betaMat[currIngredientCounter, currLayerComponentIdx]
                if currIngredientCounter == 0:
                    # We are at the start
                    fusedModelStateDict[currLayer]  = currModelStateDict[currLayer].cpu().clone() * beta
                else:
                    fusedModelStateDict[currLayer] = fusedModelStateDict[currLayer].cpu().clone() + currModelStateDict[currLayer].cpu().clone() * beta
        return fusedModelStateDict

    # Convert the 'optimal' lambdas (lambaAll[j][i]) to the equivalent 'betas'.
    # See the 'schmierzettl' image in directory 'documentation'.
    # See also my engineering notes.
    # Note 'lambaAll[j]' is the result of the optimization for the
    # combination of A with the 'j + 1'-th ingredient C
    # So the number of rows in 'lambdaAll' is 'numberIngredients - 1'
    # If we have only one ingredient, then lambdaAll is 'None'
    def convertLambdaToBeta(self, lambaAll):
        # Get # of ingredients (for model soup) and # of model components
        beta = None
        numberIngredients = len(self.ingredientIndices)    
        if numberIngredients == 1:
            # Only one ingredient (the first sorted model) -> set all factors in 'beta' to '1.0'
            beta = np.ones(shape = (1, self.numberComponents))
        else:
            # 'lambda' is a numpy array 'lambda[j, i]' with j in range [0, #ingredients - 1] and i in range [0, #model_components]
            # The returned 'betas' are used in the second phase as 'start values'.
            # What we do here is to 'unwrap' the lambdas into the equivalent 'betas'.
            beta = np.zeros(shape = (numberIngredients, self.numberComponents))
            # Set 'k' (see 'schmierzettl'), check for special case when we have only ONE component
            k = numberIngredients - 2
            # Calculate 'p[j,...]' and 's[j, ...]' (see schmierzettl)
            p = np.zeros_like(beta)
            s = np.zeros_like(beta)
            for j in range(0, numberIngredients):
                for i in range(0, self.numberComponents):
                    # 'p'
                    product = 1.0
                    for idx in range (j, (k) + 1):
                        product = product * lambaAll[idx, i]
                    p[j, i] = product
                    # 's'
                    if j > 0:
                        s[j, i] = 1.0 - lambaAll[j - 1, i]
                    else:
                        s[j, i] = 1.0
                    # now calculate beta
                    beta[j, i] = p[j, i] * s[j, i]
        # Alright, we ar edone
        return beta
    
    # Loads (deserializes) the results of phase 1 from a JSON file
    # Note it first looks locally in the 'run' directory.
    # If it's not found there, it then looks also in the 'manifoldmixms' root directory.
    def loadResultsFromJson(self, resultsFileToLoad):
        # See https://stackoverflow.com/questions/41476636/how-to-read-a-json-file-and-return-as-dictionary-in-python
        resultDict = dict()
        with open(resultsFileToLoad + '.json') as f_in:
            resultDict = json.load(f_in)
        self.numberComponents = resultDict['numberComponents']
        self.ingredientIndices = resultDict['ingredientIndices']
        self.indicesNearMisses = resultDict['indicesNearMisses']
        self.beta = JrsUtilMisc.getNumpy2DArrayFromDict(resultDict['beta'])
        self.beta = self.beta.astype(np.float32)
        # We constructed the fused model (finall accumulator model A) from 'beta' and 'ingredientIndices'
        self.fusedModelStateDict = self.getFusedModelStateDictFromBeta(self.ingredientIndices, self.beta)

    # Saves (serializes) the results of phase 1 into a JSON file, in run dir (usually)
    def saveResultsToJson(self):
        self.ctx.logger.info(f"Saving results of phase 1 to file: {self.ctx.cfg.phase1.result_filename} ")
        resultDict = {
            'numberComponents': self.numberComponents,
            'ingredientIndices': self.ingredientIndices,
            'indicesNearMisses': self.indicesNearMisses,
            'beta': JrsUtilMisc.getDictFromNumpy2DArray(self.beta),
            # Note 'self.fusedModelStateDict' is not saved, as we can re-construct it from 'beta' and 'ingedientIndices'
        }
        # See https://stackoverflow.com/questions/17043860/how-to-dump-a-dict-to-a-json-file
        with open(self.ctx.cfg.phase1.result_filename + '.json', 'w') as jsonFile:
            json.dump(resultDict, jsonFile, indent = 2)


###
### ( ) Helper functions used in phase 1
###


### TODO
