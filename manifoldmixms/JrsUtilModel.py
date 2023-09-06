# FAH This file was added by me (Hannes Fassold, JOANNEUM RESEARCH).
# It provides functions for handling the _parameters_ of Pytorch models

# ordered dictionary
from collections import OrderedDict

# Python Regex library
import re

import torch
import numpy

def printModelStructure(base_model, mode = 0):
    # FAH get some information about the model structure (its blocks and layers)
    print (base_model)
    #print ("\n\n ================= \n\n")
    #for layer_name, layer in base_model.named_modules():
    #    print(layer_name)
    #print("\n\n ================= \n\n")
    # via state_dict (I think this is the way to go)

def printModelStateDictKeys(model, mode = 0):
    if mode == 0:
        for key in model.state_dict():
            value = model.state_dict()[key]
            print(key)
    else:
        keysAll = model.state_dict().keys()
        print(keysAll)

def printModelNamedParameters(model, mode = 0):
    # For difference between state dict keys and named parameters, see:
    # See https://discuss.pytorch.org/t/difference-between-state-dict-and-parameters/37531
    # and https://stackoverflow.com/questions/54746829/pytorch-whats-the-difference-between-state-dict-and-parameters
    for name, param in model.named_parameters():
        print(name)

# This is a function which groups the state dict keys of a model by their 'level' into differenct 'components'
# It returns a dictionary where each key is the component name, and its value is a list
# containing _all_ state dict keys which are grouped 'under' this component.
# This method is very useful to get the 'building blocks' (up to a certain level) of a neural network
# - The 'default_level' specifies how coarse / fine you want to have your grouping.
#   When you set it to zero, you only get the immediate children of the model (a very coarse grouping).
#   Set it to 1 to get also the children of the children (so you have a finer grouping), etc.
# -  The 'specialCases' parameter can be set to certain components for which you want to set
#    _different_ level than the default level.
#    For each special case, add a (key, value) pair in the following way:
#    - Set the key to the string prefix of the component
#    - Set the value to the desired level for this component
#    Note in the string for the key, you can set a _placeholders_ / wildcard for an integer value via the '<INT>' placeholder.
#    This is often useful, often building blocks are enumerated in the way '<component>.<index>' with index in a certain range
#    These will NOT be grouped together, but will be seen and returned as _separate_ components
#    The alternative would be to explicitly add alll of them in 'specialCases' dictionary, which is tiresome.
#    Currently, only one '<INT>' wildcard is accepted, and it must be at the _end_ of the component string.
# Note: The functions assumes that each level is specified by a '.' in the name of the respective state dict key.
# Note: For convenience, the function returns also a dictionary mapping each component to an index (starting with zero)
# Note: For safety, both returned dictionaries are of type 'OrderedDict' (they preserve the order of insertion)
# Example call for a CLIP ViT-B/32 model (we will get 36 components then):
# components, componentsIndices = groupStateDictKeysByLevel(model.state_dict(), 1,
#              specialCases = { 'visual.transformer.resblocks.<INT>' : 3, 'transformer.resblocks.<INT>' : 2 })
def groupStateDictKeysByLevel(modelStateDict, default_level = 0, specialCases = dict()):
    resultDictComponents = OrderedDict()
    resultDictComponentsIndices = OrderedDict()
    currIdx = 0
    for keyStr in modelStateDict:
        # value = model.state_dict()[key]
        # We assume that each level is separated via '.' character in the state dict key name
        keyTokens = keyStr.split(".")
        # Determine the level to use for this key
        levelToUse = default_level
        # level to use cannot be higher than (number of tokens in 'keyStr') - 1)
        levelToUse = min(levelToUse, len(keyTokens) - 1)

        # Check the special cases
        for specialCaseKey in specialCases:
            # Note if a wildcard is involved, 'specialCasePrefix' will differ from 'specialCaseKey'
            specialCasePrefix = specialCaseKey
            specialCaseLevelToUse = specialCases[specialCaseKey]
            # level to use cannot be higher than (number of tokens in 'keyStr') - 1)
            specialCaseLevelToUse = min(specialCaseLevelToUse, len(keyTokens) - 1)
            specialCaseWildCardIntegerIndex = -1

            # Check if we have a wildcard ('<INT>') at end (and the 'key' string has an integer number at the end).
            # If so, we replace the wildcard with the respective integer number at the same level in 'keyStr'
            if specialCaseKey.endswith('<INT>'):
                # Extract the _last_ integer number from 'keyStr'
                # See https://stackoverflow.com/questions/4289331/how-to-extract-numbers-from-a-string-in-python
                numbersInKeyStr = re.findall(r'\d+', keyStr)
                if len(numbersInKeyStr) >= 1:
                    # We have at least one number in key str. Extract the last number.
                    lastNumberinKeyStr = numbersInKeyStr[-1]
                    # Generate 'keyStrUpToSpecifiedLevel', which has the number of levels specified in 'specialCaseLevelToUse'
                    keyTokensItemsUpToSpecifiedLevel = keyTokens[0 : specialCaseLevelToUse + 1]
                    keyStrUpToSpecifiedLevel = ".".join(keyTokensItemsUpToSpecifiedLevel)
                    # Check if 'keyStrUpToSpecifiedLevel' has an (integer) number at end
                    # Taken from https://stackoverflow.com/questions/14471177/python-check-if-the-last-characters-in-a-string-are-numbers
                    keyStrUpToSpecifiedLevelHasNumberAtEnd = False
                    numberAtEndRegexResult = re.search(r'\d+$', keyStrUpToSpecifiedLevel)
                    if numberAtEndRegexResult is not None:
                        # We now _replace_ the wildcard '*' in 'specialCasePrefix' with the last number from 'keyStrUpToSpecifiedLevelHasNumberAtEnd'
                        keyStrUpToSpecifiedLevelLastNumber = re.findall(r'\d+', keyStrUpToSpecifiedLevel)[-1]
                        specialCasePrefix = specialCasePrefix.replace("<INT>", keyStrUpToSpecifiedLevelLastNumber)
            # ( ) Check whether the current special case is to be used for 'keyStr'
            # If so, then we modify 'levelToUse' and quite the loop over the special cases
            if keyStr.startswith(specialCasePrefix):
                levelToUse = specialCaseLevelToUse
                break

        # ( ) We now have determined the final level to use
        # Extract the specific prefix (up to the specified level) from 'keyStr', this will be the dict key.
        keyTokensItemsUpToSpecifiedLevel = keyTokens[0: levelToUse + 1]
        keyStrUpToSpecifiedLevel = ".".join(keyTokensItemsUpToSpecifiedLevel)
        # If respective key does not exist in 'resultDictComponents', create it and set its value to empty list
        if not keyStrUpToSpecifiedLevel in resultDictComponents:
            resultDictComponentsIndices[keyStrUpToSpecifiedLevel] = len(resultDictComponents)
            resultDictComponents[keyStrUpToSpecifiedLevel] = [ ]
        # Now append to the list
        resultDictComponents[keyStrUpToSpecifiedLevel].append(keyStr)

    # We are done with all items in models 'state_dict'
    return resultDictComponents, resultDictComponentsIndices

# get component info for a certain layer
# The parameters components, componentsIndices are the result of fn. 'groupStateDictKeysByLevel'
def getComponentInfoForLayer(layerName, components, componentsIndices):
    for component in components:
        componentLayers = components[component]
        if layerName in componentLayers:
            # Alright, we found the layer in the current component. Return its key and index
            return component, componentsIndices[component]
    # We did not find the layer in any of the components
    return '<<ERR_LAYER_NOT_FOUND>>', -1
             


# Note this function is specially for a model.
# Curently, only the CLIP model is supported.
# Note 'modelStateDict' should be the _actual_ stateDict of a real 'full' model (which can be used for inference)
def getComponentsFromComponentGranularity(modelStateDict, modelName, componentGranularity):
    components = None
    if modelName == 'ViT-B/32':
        if componentGranularity == 0:
            components = groupStateDictKeysByLevel(modelStateDict, 0)
        elif componentGranularity == 1:
            components = groupStateDictKeysByLevel(modelStateDict, 1)
        elif componentGranularity == 2:
            components = groupStateDictKeysByLevel(modelStateDict, 2)
        elif componentGranularity == 3:
            components = groupStateDictKeysByLevel(modelStateDict, 2,
                            specialCases = { 'model.visual.transformer.resblocks.<INT>' : 4, 'model.transformer.resblocks.<INT>' : 3 })
        elif componentGranularity == 1000:
            components = groupStateDictKeysByLevel(modelStateDict, 1000)
    else:
        raise ValueError('Cannot extract components for this model architecture - currently only CLIP model is supported')
    return components
