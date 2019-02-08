# -*- coding: utf-8 -*-
#!/usr/bin/env python2
'''
Based on code by kyubyong park at https://www.github.com/kyubyong/dc_tts
'''
from __future__ import print_function

import os
import sys
import glob
import shutil
import random
from argparse import ArgumentParser

import numpy as np
import tensorflow as tf

from architectures import Text2MelGraph, SSRNGraph, BabblerGraph
from data_load import load_data
from synthesize import synth_text2mel, synth_mel2mag, split_batch, make_mel_batch
from objective_measures import compute_dtw_error, compute_simple_LSD
from libutil import basename, safe_makedir, load_config

import logger_setup
from logging import info

from tqdm import tqdm



def compute_validation(hp, model_type, epoch, inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_set_reference):
    if model_type == 't2m':
        validation_set_predictions_tensor, lengths = synth_text2mel(hp, inputs, synth_graph, sess, speaker_data=speaker_codes)
        validation_set_predictions = split_batch(validation_set_predictions_tensor, lengths)  
        score = compute_dtw_error(validation_set_reference, validation_set_predictions)   
    elif model_type == 'ssrn':
        validation_set_predictions_tensor = synth_mel2mag(hp, inputs, synth_graph, sess)
        lengths = [len(ref) for ref in validation_set_reference]
        validation_set_predictions = split_batch(validation_set_predictions_tensor, lengths)  
        score = compute_simple_LSD(validation_set_reference, validation_set_predictions)
    ## store parameters for later use:-
    valid_dir = '%s-%s/validation_epoch_%s'%(hp.logdir, model_type, epoch)
    safe_makedir(valid_dir)
    for i in range(hp.validation_sentences_to_synth_params):  ### TODO: configure this
        np.save(os.path.join(valid_dir, basename(valid_filenames[i])), validation_set_predictions[i])
    return score

 

def main_work():

    #################################################
            
    # ============= Process command line ============
    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-m', dest='model_type', required=True, choices=['t2m', 'ssrn', 'babbler'])
    opts = a.parse_args()
    
    # ===============================================
    model_type = opts.model_type
    hp = load_config(opts.config)
    logdir = hp.logdir + "-" + model_type 
    logger_setup.logger_setup(logdir)
    info('Command line: %s'%(" ".join(sys.argv)))


    ### Prepare reference data for validation set:  ### TODO: alternative to holding in memory?
    if hp.multispeaker:
        (valid_filenames, validation_text, speaker_codes) = load_data(hp, mode="validation", get_speaker_codes=True)
    else:
        (valid_filenames, validation_text) = load_data(hp, mode="validation")
        speaker_codes = None  ## default


    ## take random subset of validation set to avoid 'This is a librivox recording' type sentences
    random.seed(1234)
    v_indices = range(len(valid_filenames))
    random.shuffle(v_indices)
    v = min(hp.validation_sentences_to_evaluate, len(valid_filenames))
    v_indices = v_indices[:v]

    if hp.multispeaker: ## now come back to this after v computed
        speaker_codes = np.array(speaker_codes)[v_indices].reshape(-1, 1)

    valid_filenames = np.array(valid_filenames)[v_indices]
    validation_mags = [np.load(hp.full_audio_dir + os.path.sep + basename(fpath)+'.npy') \
                                for fpath in valid_filenames]                                
    validation_text = validation_text[v_indices, :]

    if model_type=='t2m':
        validation_mels = [np.load(hp.coarse_audio_dir + os.path.sep + basename(fpath)+'.npy') \
                                    for fpath in valid_filenames]
        validation_inputs = validation_text
        validation_reference = validation_mels
        validation_lengths = None
    elif model_type=='ssrn':
        validation_inputs, validation_lengths = make_mel_batch(hp, valid_filenames)
        validation_reference = validation_mags

    ## Map to appropriate type of graph depending on model_type:
    AppropriateGraph = {'t2m': Text2MelGraph, 'ssrn': SSRNGraph, 'babbler': BabblerGraph}[model_type]

    g = AppropriateGraph(hp) ; info("Training graph loaded")
    synth_graph = AppropriateGraph(hp, mode='synthesize', reuse=True) ; info("Synthesis graph loaded")

    if 0:
        print (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel'))
        ## [<tf.Variable 'Text2Mel/TextEnc/embed_1/lookup_table:0' shape=(61, 128) dtype=float32_ref>, <tf.Variable 'Text2Mel/TextEnc/C_2/conv1d/kernel:0' shape=(1, 128, 512) dtype=float32_ref>, ...

    ## TODO: tensorflow.python.training.supervisor deprecated: --> switch to tf.train.MonitoredTrainingSession  
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)

    ## If save_every_n_epochs > 0, models will be stored here every n epochs and not
    ## deleted, regardless of validation improvement etc.:--
    safe_makedir(logdir + '/archive/')

    with sv.managed_session() as sess:
        
        if hp.restart_from_savepath: 
            info('Restart from these paths:')
            info(hp.restart_from_savepath)
            
            assert len(hp.restart_from_savepath) == 2
            restart_from_savepath1, restart_from_savepath2 = hp.restart_from_savepath
            restart_from_savepath1 = os.path.abspath(restart_from_savepath1)
            restart_from_savepath2 = os.path.abspath(restart_from_savepath2)

            sess.graph._unsafe_unfinalize() ## !!! https://stackoverflow.com/questions/41798311/tensorflow-graph-is-finalized-and-cannot-be-modified/41798401
            sess.run(tf.global_variables_initializer())

            print ('Restore parameters')
            if model_type == 't2m':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
                saver1 = tf.train.Saver(var_list=var_list)
                saver1.restore(sess, restart_from_savepath1)
                print("Text2Mel Restored!")
            elif model_type == 'ssrn':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
                saver2 = tf.train.Saver(var_list=var_list)
                saver2.restore(sess, restart_from_savepath2)
                print("SSRN Restored!")


        loss_history = []
        epoch = 0  ## TODO: this counter won't work if training restarts in same directory. 
                   ## Get epoch from gs? 

        current_score = compute_validation(hp, model_type, epoch, validation_inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_reference)
        info('validation epoch {0}: {1:0.3f}'.format(epoch, current_score))
            
        while 1:            
            progress_bar_text = '%s/%s; ep. %s'%(hp.config_name, model_type, epoch)
            for step_in_current_epoch in tqdm(range(g.num_batch), total=g.num_batch, ncols=80, leave=True, unit='b', desc=progress_bar_text):
                gs, loss_components, _ = sess.run([g.global_step, g.loss_components, g.train_op])
                loss_history.append(loss_components)

            ### End of epoch: validate?
            if hp.validate_every_n_epochs:
                if epoch % hp.validate_every_n_epochs == 0:
                    
                    loss_history = np.array(loss_history)
                    train_loss_mean_std = np.concatenate([loss_history.mean(axis=0), loss_history.std(axis=0)])
                    loss_history = []

                    train_loss_mean_std = ' '.join(['{:0.3f}'.format(score) for score in train_loss_mean_std])
                    info('train epoch {0}: {1}'.format(epoch, train_loss_mean_std))

                    current_score = compute_validation(hp, model_type, epoch, validation_inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_reference)
                    info('validation epoch {0:0}: {1:0.3f}'.format(epoch, current_score))
                    
            ### Save end of each epoch (all but the most recent 5 will be overwritten):       
            stem = logdir + '/model_epoch_{0}'.format(epoch)
            sv.saver.save(sess, stem)

            ### Check if we should archive (to files which won't be overwritten):
            if hp.save_every_n_epochs:
                if epoch % hp.save_every_n_epochs == 0:
                    info('Archive model %s'%(stem))
                    for fname in glob.glob(stem + '*'):
                        shutil.copy(fname, logdir + '/archive/')
            epoch += 1
            if epoch > hp.max_epochs: 
                info('Max epochs ({}) reached: end training'.format(hp.max_epochs)); return

    print("Done")


if __name__ == "__main__":

    main_work()
