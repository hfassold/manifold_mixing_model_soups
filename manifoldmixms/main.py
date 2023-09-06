# FAH: Note my main change to 'main_with_objectnet.py'is that I SKIP the 'ObjectNet' dataset
# It is huge, so I didn't download it ..
# The source here has been changed appropriately.

import argparse
import os
import wget
import torch
import clip
import os
import json
import operator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from datasets import ImageNet2p, ImageNet, ImageNetV2, ImageNetSketch, ImageNetR, ImageNetA
from utils import get_model_from_sd, test_model_on_dataset

# FAH Import the additional stuff needed for ManifoldMixMS
import hydra
from omegaconf import DictConfig, OmegaConf

def format_three_digits(digits: int) -> str:
    return "{:03d}".format(digits)

# FAH We register a new resolver 'jrs_format_three_digits' which formats a number so that it has always 3 digits
# (filling up with leading zero). Is used in master config file to make the run number always 3-digits
# See https://github.com/facebookresearch/hydra/issues/1795
OmegaConf.register_new_resolver("jrs_format_three_digits", resolver=format_three_digits)

# my additional python files
import JrsUtilModel
import JrsContext
import JrsPhase1
import JrsPhase2
# ~FAH

# - Note you can specifiy a config file with a _different_ name via the "--config-name" command line parameter,
#   see https://stackoverflow.com/questions/62664386/how-to-pass-a-hydra-config-via-command-line
#   See https://hydra.cc/docs/intro/ for more information regarding the Hydra package
# - version_base = None means 'take behaviour of currently used Hydra version', see https://hydra.cc/docs/upgrades/version_base/
@hydra.main(config_path=".", config_name="config_manifoldmixms", version_base = None)
def main(masterCfg: DictConfig):
    # params:s --greedy-soup --data-location ../manifoldmixms_datasets --model-location ./models

    # ( ) Create the global 'context' object.
    # Contains config vars, logger.
    # It creates also the 'base' network model, the dataset objects etcs.
    originalWorkDir = hydra.utils.get_original_cwd()
    ctx = JrsContext.Context(masterCfg, originalWorkDir)

    # Visualize one ingredient model (CLIP network)
    #ctx.visualizeOneIngredientModel()

    # ( ) Run phase 1 (or load its result)
    ctxPhase1 = JrsPhase1.Context(ctx)
    if masterCfg.do_phase1:
        ctxPhase1.run()
    else:
        phase1ResultFileName = f'{ctx.originalWorkDir}/{ctx.cfg.phase2.phase1_result_filename}'
        ctx.logger.info(f'Loading phase 1 results from file: {phase1ResultFileName} ')
        ctxPhase1.loadResultsFromJson(phase1ResultFileName)
    fusedModelPhase1 = ctx.getModelFromStateDict(ctxPhase1.fusedModelStateDict)
    accuracyPhase1ForDatasetDict = ctx.evalModelOnDatasetsAndPrint(fusedModelPhase1, 'fusedModelPhase1')

    # ( ) Run phase 2 (or load its result)
    if masterCfg.do_phase2:
        ctxPhase2 = JrsPhase2.Context(ctx, ctxPhase1)
        ctxPhase2.run()

    # Step 5: Plot.
    if True:
        individual_model_db = pd.read_json(ctx.individualModelsResultFile, lines=True)
        individual_model_db['OOD'] = 1./5 * (individual_model_db['ImageNetV2'] +
            individual_model_db['ImageNetR'] + individual_model_db['ImageNetSketch'] +
            individual_model_db['ObjectNet'] +
            individual_model_db['ImageNetA'])
        uniform_soup_db = pd.read_json(f'{ctx.originalWorkDir}/uniform_soup_results.jsonl', lines=True)
        uniform_soup_db['OOD'] = 1./5 * (uniform_soup_db['ImageNetV2'] +
            uniform_soup_db['ImageNetR'] + uniform_soup_db['ImageNetSketch'] +
            uniform_soup_db['ObjectNet'] +
            uniform_soup_db['ImageNetA'])
        greedy_soup_db = pd.read_json(f'{ctx.originalWorkDir}/greedy_soup_results.jsonl', lines=True)
        greedy_soup_db['OOD'] = 1./5 * (greedy_soup_db['ImageNetV2'] +
            greedy_soup_db['ImageNetR'] + greedy_soup_db['ImageNetSketch'] +
            greedy_soup_db['ObjectNet'] +
            greedy_soup_db['ImageNetA'])
        # FAH Add entry for manifoldmixms algorithm. With the default yaml config, we have loaded ManifoldMixMS-C8 variant with 8 components.
        manifoldmixms_soup_db = dict()
        manifoldmixms_soup_db['ImageNet'] = accuracyPhase1ForDatasetDict['ImageNet']
        manifoldmixms_soup_db['OOD'] = accuracyPhase1ForDatasetDict['average_out_of_dist']

        fig = plt.figure(constrained_layout=True, figsize=(8, 6))
        ax = fig.subplots()

        ax.scatter(
            manifoldmixms_soup_db['ImageNet'],
            manifoldmixms_soup_db['OOD'],
            marker='*',
            color='C3',
            s=350,
            label='ManifoldMixMS-C8',
            zorder=10
        )

        ax.scatter(
            greedy_soup_db['ImageNet'], 
            greedy_soup_db['OOD'],
            marker='o',
            #marker='*',
            color='C4',
            s=180,
            label='Greedy Soup',
            zorder=10
        )

        ax.scatter(
            uniform_soup_db['ImageNet'], 
            uniform_soup_db['OOD'], 
            marker='o',
            color='C0',
            s=180,
            label='Uniform Soup',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[0], 
            individual_model_db['OOD'].values[0], 
            marker='h',
            color='slategray',
            s=150,
            label='Initialization (LP)',
            zorder=10
        )

        ax.scatter(
            individual_model_db['ImageNet'].values[1:], 
            individual_model_db['OOD'].values[1:], 
            marker='d',
            color='C2',
            s=110,
            label='Various hyperparameters',
            zorder=10
        )

        # FAH: Originally, it was fontsize 16 for x- and y-label
        ax.set_ylabel('Avg. accuracy on 5 distribution shifts', fontsize=13)
        ax.set_xlabel('ImageNet Accuracy (top-1)', fontsize=13)
        ax.grid()
        ax.legend(fontsize=13)
        plt.savefig(f'{ctx.originalWorkDir}/figure_manifoldmixms.png', bbox_inches='tight')


if __name__ == "__main__":
    main()