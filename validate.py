# -*- coding: utf-8 -*-
#!/usr/bin/env python2
'''
Based on code by kyubyong park at https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function

SEED_VALUE=1111
import os
os.environ['PYTHONHASHSEED']=str(SEED_VALUE)
import sys
import glob
import shutil
import random
random.seed(SEED_VALUE)
from argparse import ArgumentParser

import numpy as np
np.random.seed(SEED_VALUE)
import tensorflow as tf
tf.set_random_seed(SEED_VALUE)
from tensorflow.python import debug as tf_debug
import time
from architectures import Text2MelGraph, SSRNGraph, BabblerGraph
from data_load import load_data
from synthesize import synth_text2mel, synth_mel2mag, split_batch, make_mel_batch, synth_codedtext2mel, get_text_lengths, encode_text, list2batch, restore_archived_model_parameters
from objective_measures import compute_dtw_error, compute_simple_LSD, plot_objective_measures
from libutil import basename, safe_makedir
from configuration import load_config
from utils import plot_alignment, plot_loss_history
import math
from utils import durations_to_position, end_pad_for_reduction_shape_sync

import logger_setup
from logging import info

from tqdm import tqdm

def compute_validation(hp, model_type, inputs, synth_graph, sess, speaker_codes, \
         valid_filenames, validation_set_reference, duration_data=None, validation_labels=None, position_in_phone_data=None):
    if model_type == 't2m': ## TODO: coded_text2mel here
        validation_set_predictions_tensor, lengths = synth_text2mel(hp, inputs, synth_graph, sess, speaker_data=speaker_codes, duration_data=duration_data, labels=validation_labels, position_in_phone_data=position_in_phone_data)
        validation_set_predictions = split_batch(validation_set_predictions_tensor, lengths)
        score = compute_dtw_error(validation_set_reference, validation_set_predictions)

    ## store parameters for later use:-
    valid_dir = '%s-%s'%(hp.logdir, model_type)
    safe_makedir(valid_dir)
    hp.validation_sentences_to_synth_params = min(hp.validation_sentences_to_synth_params, len(valid_filenames)) #if less sentences match the validation pattern than the value of 'hp.validation_sent_to_synth'
    for i in range(hp.validation_sentences_to_synth_params):
        np.save(os.path.join(valid_dir, basename(valid_filenames[i])), validation_set_predictions[i])
    return score

def main_work():

    #################################################

    # ============= Process command line ============
    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-m', dest='model_type', required=True, choices=['t2m', 'ssrn', 'babbler'])
    a.add_argument('-speaker', dest='valid_speaker', required=False)
    opts = a.parse_args()

    # ===============================================
    model_type = opts.model_type
    valid_speaker = opts.valid_speaker
    hp = load_config(opts.config)

    logdir = hp.logdir + "-" + model_type
    logger_setup.logger_setup(logdir)
    info('Random seed: '+str(SEED_VALUE))
    info('VALIDATION LOG VALIDATION LOG')
    info('Command line: %s'%(" ".join(sys.argv)))

    # can we print the hyperparameters for reproducibility?
    info('hyperparameters: '+str(hp.__dict__.items()))

    ### TODO: move this to its own function somewhere. Can be used also at synthesis time?
    ### Prepare reference data for ation set:  ### TODO: alternative to holding in memory?
    dataset, char2idx = load_data(hp, mode="external_validation")
    valid_filenames, validation_text = dataset['fpaths'], dataset['texts']

    speaker_codes = validation_duration_data = position_in_phone_data = None ## defaults

    if hp.multispeaker: speaker_codes = dataset['speakers']

    ## take random subset of validation set to avoid 'This is a librivox recording' type sentences
    v_indices = range(len(valid_filenames))
    v = min(hp.validation_sentences_to_evaluate, len(valid_filenames))
    v_indices = v_indices[:v]

    if hp.multispeaker: ## now come back to this after v computed
        speaker_codes = np.array(speaker_codes)[v_indices].reshape(-1, 1)
    if hp.use_external_durations:
        validation_duration_data = validation_duration_data[v_indices, :, :]


    valid_filenames = np.array(valid_filenames)[v_indices]
    for files in valid_filenames:
        info('validation files: '+str(files)) # to know which files we are validating
    validation_mags = [np.load(hp.full_audio_dir + os.path.sep + basename(fpath)+'.npy') \
                                for fpath in valid_filenames]

    validation_text = validation_text[v_indices, :]
    validation_labels = None # default
    validation_mels = [np.load(hp.coarse_audio_dir + os.path.sep + basename(fpath)+'.npy') \
                                for fpath in valid_filenames]
    validation_inputs = validation_text
    validation_reference = validation_mels
    validation_lengths = None

    if hp.multispeaker: ## here try different validation ways?
        validation_speaker = np.ones((hp.validation_sentences_to_evaluate, 1))
        # If validation script declares a specific speaker
        if valid_speaker:
            info('Using speaker: '+str(valid_speaker))
            for i in range(0, len(valid_filenames[:hp.validation_sentences_to_evaluate])):
                validation_spkr_id = valid_speaker
                speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))
                speaker_ix = speaker2ix[validation_spkr_id]
                validation_speaker[i] = validation_speaker[i] * speaker_ix

        # Use a map of speakers
    elif speaker_maps_valid in hp:
            info('Using speaker maps')
            for i in range(0, len(valid_filenames[:hp.validation_sentences_to_evaluate])):
                validation_spkr_id = valid_filenames[i].split('/')[-1].replace('dev_', '').split('_')[0]
                validation_spkr_id = speaker_maps_valid[validation_spkr_id]
                speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))
                speaker_ix = speaker2ix[validation_spkr_id]
                validation_speaker[i] = validation_speaker[i] * speaker_ix

        # Use speakers in the id of the files
        else:
            info('Using speaker ids in file names')
            for i in range(0, len(valid_filenames[:hp.validation_sentences_to_evaluate])):
                validation_spkr_id = valid_filenames[i].split('/')[-1].replace('dev_', '').split('_')[0]
                speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))
                speaker_ix = speaker2ix[validation_spkr_id]
                validation_speaker[i] = validation_speaker[i] * speaker_ix


    # accumulate validation scores
    acc_validation_scores = []
    validation_epochs = []

    ## Map to appropriate type of graph depending on model_type:
    g = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in range(0, hp.max_epochs, hp.validate_every_n_epochs):
            t2m_epoch = restore_archived_model_parameters(sess, hp, 't2m', epoch)
            current_score = compute_validation(hp, model_type, validation_inputs, g, sess, speaker_codes, valid_filenames, validation_reference, duration_data=validation_duration_data, validation_labels=validation_labels, position_in_phone_data=position_in_phone_data)
            acc_validation_scores.append(current_score)
            validation_epochs.append(epoch)
            info('validation epoch {0}: {1:0.3f}'.format(epoch, current_score))

    info("acc_validation_scores: "+acc_validation_scores)
    info("validation_epochs: "+validation_epochs)
    info("Done")


if __name__ == "__main__":

    main_work()
