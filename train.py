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
from tensorflow.python import debug as tf_debug

from architectures import Text2MelGraph, SSRNGraph, BabblerGraph
from data_load import load_data
from synthesize import synth_text2mel, synth_mel2mag, split_batch, make_mel_batch, synth_codedtext2mel, get_text_lengths, encode_text
from objective_measures import compute_dtw_error, compute_simple_LSD
from libutil import basename, safe_makedir
from configuration import load_config
from utils import plot_alignment

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
    else:
        info('compute_validation cannot handle model type %s: dummy value (0.0) supplied as validation score'%(model_type)); return 0.0
    ## store parameters for later use:-
    valid_dir = '%s-%s/validation_epoch_%s'%(hp.logdir, model_type, epoch)
    safe_makedir(valid_dir)
    for i in range(hp.validation_sentences_to_synth_params):  ### TODO: configure this
        np.save(os.path.join(valid_dir, basename(valid_filenames[i])), validation_set_predictions[i])
    return score


def get_and_plot_alignments(hp, epoch, attention_graph, sess, attention_inputs, attention_mels, alignment_dir):
    return_values = sess.run([attention_graph.alignments], # use attention_graph to obtain attention maps for a few given inputs and mels
                             {attention_graph.L: attention_inputs, 
                              attention_graph.mels: attention_mels}) 
    alignments = return_values[0] # sess run returns a list, so unpack this list
    for i in range(hp.num_sentences_to_plot_attention):
        plot_alignment(hp, alignments[i], i+1, epoch, dir=alignment_dir)

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
    dataset = load_data(hp, mode="validation") 
    valid_filenames, validation_text = dataset['fpaths'], dataset['texts']
    if hp.multispeaker:
        speaker_codes = dataset['speakers']
    else:
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
    else:
        info('Undefined model_type {} for making validation inputs -- supply dummy None values'.format(model_type))
        validation_inputs = None
        validation_reference = None

    ## Get the text and mel inputs for the utts you would like to plot attention graphs for 
    if hp.plot_attention_every_n_epochs and model_type=='t2m': #check if we want to plot attention
        # TODO do we want to generate and plot attention for validation or training set sentences??? modify attention_inputs accordingly...
        attention_inputs = validation_text[:hp.num_sentences_to_plot_attention]
        attention_mels = validation_mels[:hp.num_sentences_to_plot_attention]
        attention_mels = np.array(attention_mels) #TODO should be able to delete this line...?        
        attention_mels_array = np.zeros((hp.num_sentences_to_plot_attention, hp.max_T, hp.n_mels), np.float32) # create fixed size array to hold attention mels
        for i in range(hp.num_sentences_to_plot_attention): # copy data into this fixed sized array
            attention_mels_array[i, :attention_mels[i].shape[0], :attention_mels[i].shape[1]] = attention_mels[i]
        attention_mels = attention_mels_array # rename for convenience

    ## Map to appropriate type of graph depending on model_type:
    AppropriateGraph = {'t2m': Text2MelGraph, 'ssrn': SSRNGraph, 'babbler': BabblerGraph}[model_type]

    g = AppropriateGraph(hp) ; info("Training graph loaded")
    synth_graph = AppropriateGraph(hp, mode='synthesize', reuse=True) ; info("Synthesis graph loaded") #reuse=True ensures that 'synth_graph' and 'attention_graph' share weights with training graph 'g'
    attention_graph = AppropriateGraph(hp, mode='generate_attention', reuse=True) ; info("Atttention generating graph loaded")
    #TODO is loading three graphs a problem for memory usage?

    if 0:
        print (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel'))
        ## [<tf.Variable 'Text2Mel/TextEnc/embed_1/lookup_table:0' shape=(61, 128) dtype=float32_ref>, <tf.Variable 'Text2Mel/TextEnc/C_2/conv1d/kernel:0' shape=(1, 128, 512) dtype=float32_ref>, ...

    ## TODO: tensorflow.python.training.supervisor deprecated: --> switch to tf.train.MonitoredTrainingSession  
    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)

    ## Get the current training epoch from the name of the model that we have loaded
    latest_checkpoint = tf.train.latest_checkpoint(logdir)
    if latest_checkpoint:
        epoch = int(latest_checkpoint.strip('/ ').split('/')[-1].replace('model_epoch_', ''))
    else: #did not find a model checkpoint, so we start training from scratch
        epoch = 0

    ## If save_every_n_epochs > 0, models will be stored here every n epochs and not
    ## deleted, regardless of validation improvement etc.:--
    safe_makedir(logdir + '/archive/')

    with sv.managed_session() as sess:
        if 0:  ## Set to 1 to debug NaNs; at tfdbg prompt, type:    run -f has_inf_or_nan 
            ## later:    lt  -f has_inf_or_nan -n .*AudioEnc.*
            os.system('rm -rf {}/tmp_tfdbg/'.format(logdir))
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, dump_root=logdir+'/tmp_tfdbg/')       

        if hp.initialise_weights_from_existing:
            info('=====Initialise some variables from existing model(s)=====') 
            sess.graph._unsafe_unfinalize() ## !!! https://stackoverflow.com/questions/41798311/tensorflow-graph-is-finalized-and-cannot-be-modified/41798401
            for (scope, checkpoint) in hp.initialise_weights_from_existing:
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
                info('----From existing model %s:----'%(checkpoint))  
                if var_list: ## will be empty when training t2m but looking at ssrn
                    saver = tf.train.Saver(var_list=var_list)
                    saver.restore(sess, checkpoint)         
                    for var in var_list:
                        info('   %s'%(var.name))    
                else:
                    info('   No variables!')   
                info('========================================================') 
             
        if hp.restart_from_savepath: #set this param to list: [path_to_t2m_model_folder, path_to_ssrn_model_folder]
            # info('Restart from these paths:')
            info(hp.restart_from_savepath)
            
            # assert len(hp.restart_from_savepath) == 2
            restart_from_savepath1, restart_from_savepath2 = hp.restart_from_savepath
            restart_from_savepath1 = os.path.abspath(restart_from_savepath1)
            restart_from_savepath2 = os.path.abspath(restart_from_savepath2)

            sess.graph._unsafe_unfinalize() ## !!! https://stackoverflow.com/questions/41798311/tensorflow-graph-is-finalized-and-cannot-be-modified/41798401
            sess.run(tf.global_variables_initializer())

            print ('Restore parameters')
            if model_type == 't2m':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
                saver1 = tf.train.Saver(var_list=var_list)
                latest_checkpoint = tf.train.latest_checkpoint(restart_from_savepath1)
                saver1.restore(sess, restart_from_savepath1)
                print("Text2Mel Restored!")
            elif model_type == 'ssrn':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
                saver2 = tf.train.Saver(var_list=var_list)
                latest_checkpoint = tf.train.latest_checkpoint(restart_from_savepath2)
                saver2.restore(sess, restart_from_savepath2)
                print("SSRN Restored!")
            epoch = int(latest_checkpoint.strip('/ ').split('/')[-1].replace('model_epoch_', ''))
            # TODO: this counter won't work if training restarts in same directory.
            ## Get epoch from gs?
        
        loss_history = [] #any way to restore loss history too?

        #plot attention generated from freshly initialised model
        if hp.plot_attention_every_n_epochs and model_type == 't2m' and epoch == 0: # ssrn model doesn't generate alignments 
            get_and_plot_alignments(hp, epoch - 1, attention_graph, sess, attention_inputs, attention_mels, logdir + "/alignments") # epoch-1 refers to freshly initialised model
 
        current_score = compute_validation(hp, model_type, epoch, validation_inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_reference)
        info('validation epoch {0}: {1:0.3f}'.format(epoch, current_score))

        while 1:  
            progress_bar_text = '%s/%s; ep. %s'%(hp.config_name, model_type, epoch)
            for batch_in_current_epoch in tqdm(range(g.num_batch), total=g.num_batch, ncols=80, leave=True, unit='b', desc=progress_bar_text):
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

            ### End of epoch: plot attention matrices? #################################
            if hp.plot_attention_every_n_epochs and model_type == 't2m' and epoch % hp.plot_attention_every_n_epochs == 0: # ssrn model doesn't generate alignments 
                get_and_plot_alignments(hp, epoch, attention_graph, sess, attention_inputs, attention_mels, logdir + "/alignments")

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
