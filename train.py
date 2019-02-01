# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from tqdm import tqdm


import glob
import shutil
from data_load import load_data
from modules import *
# from networks import TextEnc, AudioEnc, AudioDec, Attention, SSRN
import tensorflow as tf
from utils import *
import sys


## to import configs
# HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append( HERE + '/config/' )
# import importlib
import imp


from argparse import ArgumentParser

#from graphs import Graph, AudioPredictor
from architectures import Text2MelGraph, SSRNGraph, BabblerGraph


from synthesize import synth_text2mel, synth_mel2mag, split_batch, make_mel_batch
from objective_measures import compute_dtw_error, compute_simple_LSD
from libutil import basename, safe_makedir, load_config

import logger_setup
from logging import info




def compute_validation(hp, model_type, epoch, inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_set_reference):
    if model_type=='t2m':
        validation_set_predictions_tensor, lengths = synth_text2mel(hp, inputs, synth_graph, sess, speaker_data=speaker_codes)
        validation_set_predictions = split_batch(validation_set_predictions_tensor, lengths)  
        score = compute_dtw_error(validation_set_reference, validation_set_predictions)   
    elif model_type=='ssrn':
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












# import re
# def basename(fname):
#     path, name = os.path.split(fname)
#     base = re.sub('\.[^\.]+\Z','',name)
#     return base



# def synth_valid_batch(hp, sess, g, outdir, num=1):
#     L = load_data(hp, mode="validation")

#     if num==1:
#         Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
#         prev_max_attentions = np.zeros((len(L),), np.int32)

#         for j in tqdm(range(hp.max_T)):
#             _Y, _max_attentions, _alignments, = \
#                 sess.run([ g.Y, g.max_attentions, g.alignments],
#                          {g.L: L,
#                           g.mels: Y,
#                           g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop
#             Y[:, j, :] = _Y[:, j, :]
#             prev_max_attentions = _max_attentions[:, j]
#     else:
#         print ('pass!')
#         pass


    #print ('get mag...')
    # Get magnitude
    #Z = sess.run(g.Z, {g.Y: Y})    


def main_work():

    #################################################
            
    # ============= Process command line ============


    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    # a.add_argument('-m', dest='num', required=True, type=int, choices=[1, 2, 3], \
    #                 help='1: Text2mel, 2: SSRN, 3: Audio encoder/decoder only')
    a.add_argument('-m', dest='model_type', required=True, choices=['t2m', 'ssrn', 'babbler'])
    opts = a.parse_args()
    
    # ===============================================
    model_type = opts.model_type

    hp = load_config(opts.config)
    # config = os.path.abspath(opts.config)
    # assert os.path.isfile(config)
    # conf_mod = imp.load_source('config', config)
    # hp = conf_mod.Hyperparams()



    logdir = hp.logdir + "-" + model_type #str(num)
    logger_setup.logger_setup(logdir)

    info('Command line: %s'%(" ".join(sys.argv)))


    ### Prepare reference data for validation set:  ### TODO: alternative to holding in memory?
    if hp.multispeaker:
        (valid_filenames, validation_text, speaker_codes) = load_data(hp, mode="validation", get_speaker_codes=True)
    else:
        (valid_filenames, validation_text) = load_data(hp, mode="validation")
        speaker_codes = None  ## default


    ## take random subset of validation set to avoid 'This is a librivox recording' type sentences
    np.seed(1234)
    v_indices = range(len(valid_filenames))
    random.shuffle(v_indices)
    v = min(hp.validation_sentences_to_evaluate, len(valid_filenames))
    v_indices = v_indices[:v]

    if hp.multispeaker: ## now come back to this after v computed
        speaker_codes = np.array(speaker_codes[v_indices]).reshape(-1, 1) ## TODO batchsize


    valid_filenames = valid_filenames[v_indices]
    validation_mags = [np.load(hp.full_audio_dir + os.path.sep + basename(fpath)+'.npy') \
                                for fpath in valid_filenames]                                
    validation_text = validation_text[v_indices, :] ## TODO batchsize
    #validation_mags = validation_mags[:v]  ## TODO batchsize


    if model_type=='t2m':
        validation_mels = [np.load(hp.coarse_audio_dir + os.path.sep + basename(fpath)+'.npy') \
                                    for fpath in valid_filenames]
        #validation_mels = validation_mels[:hp.validation_sentences_to_evaluate]  ## TODO batchsize
        validation_inputs = validation_text
        validation_reference = validation_mels
        validation_lengths = None
    elif model_type=='ssrn':
        validation_inputs, validation_lengths = make_mel_batch(hp, valid_filenames)
        # print (type(validation_mels))
        #validation_inputs = np.array(validation_mels, dtype=np.float32)  
        validation_reference = validation_mags

    # if num in ['t2m', 'ssrn']:
    #     #g = Graph(hp, num=num); print("Training Graph loaded")
    #     #synth_graph = Graph(hp, num=num, mode='synthesize', reuse=True, separate_synthesis_graph=True) ## TODO: does the graph ever have to be shared?
    #     g = Text2MelGraph(hp) ; info("Training Graph loaded")
    #     synth_graph = Text2MelGraph(hp, mode='synthesize', reuse=True) ; info("Synthesis Graph loaded")
    # elif num=='babbler':
    #     g = AudioPredictor(hp); print("AudioPredictor training graph loaded")

    ## map to appropriate type of graph depending on model_type
    AppropriateGraph = {'t2m': Text2MelGraph, 'ssrn': SSRNGraph, 'babbler': BabblerGraph}[model_type]

    g = AppropriateGraph(hp) ; info("Training Graph loaded")
    synth_graph = AppropriateGraph(hp, mode='synthesize', reuse=True) ; info("Synthesis Graph loaded")

    if 0:
        print (tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel'))
        ## [<tf.Variable 'Text2Mel/TextEnc/embed_1/lookup_table:0' shape=(61, 128) dtype=float32_ref>, <tf.Variable 'Text2Mel/TextEnc/C_2/conv1d/kernel:0' shape=(1, 128, 512) dtype=float32_ref>, ...



    sv = tf.train.Supervisor(logdir=logdir, save_model_secs=0, global_step=g.global_step)

    # save_every = hp.save_every_n_iterations
    # archive_every = hp.archive_every_n_iterations
    # if archive_every:
    #     assert archive_every % save_every == 0
    #     if not os.path.isdir(logdir + '/archive/'):
    #         os.makedirs(logdir + '/archive/')

    #save_every = g.num_batch * float(hp.validate_n_times_per_epoch) ## validate_every_n_epochs can be fractional


    ### Pre-compute the steps (minibatch numbers) at which we will evaluate the model during training:
    # nvals = hp.validate_n_times_per_epoch
    # assert nvals > 0.0 ## TODO: config check?
    # validate_steps = np.linspace(g.num_batch/nvals, g.num_batch-1, nvals).astype(int).tolist()
    # epoch_fractions = np.linspace(1.0/nvals, 1.0, nvals).tolist()
    # validate_steps = dict(zip(validate_steps, epoch_fractions))
    # patience = hp.patience_epochs * nvals

    # info('validate at the following steps:')
    # info(validate_steps)
    
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
            if model_type=='t2m':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
                saver1 = tf.train.Saver(var_list=var_list)
                #savepath = hp.restart_from_savepath + "-1"
                #print(savepath)
                saver1.restore(sess, restart_from_savepath1)
                print("Text2Mel Restored!")
            elif model_type=='ssrn':
                var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                           tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
                saver2 = tf.train.Saver(var_list=var_list)
                #savepath = hp.restart_from_savepath + "-2"
                saver2.restore(sess, restart_from_savepath2)
                print("SSRN Restored!")



        loss_history = []
        epoch = 0  ## TODO: this counter won't work if training restarts in same directory. 
                   ## Get epoch from gs as well? 

        #fractional_epoch = 0.0
        current_score = compute_validation(hp, model_type, epoch, validation_inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_reference)
        info('validation epoch {0}: {1:0.3f}'.format(epoch, current_score))
        best_score = current_score
        checks_since_best = 0             

        while 1:
            
            progress_bar_text = '%s/%s; ep. %s'%(hp.config_name, model_type, epoch)
            for step_in_current_epoch in tqdm(range(g.num_batch), total=g.num_batch, ncols=80, leave=True, unit='b', desc=progress_bar_text):
                # if num==1:
                #     gs, losssum, lossmel, lossbd, lossatt, _ = sess.run([g.global_step, g.loss, g.loss_mels, g.loss_bd1, g.loss_att, g.train_op])
                #     loss_history.append((losssum, lossmel, lossbd, lossatt))
                # elif num==2:
                #     gs, losssum, lossmags, lossbd, _ = sess.run([g.global_step, g.loss, g.loss_mags, g.loss_bd2, g.train_op])
                #     loss_history.append((losssum, lossmags, lossbd))
                gs, loss_components, _ = sess.run([g.global_step, g.loss_components, g.train_op])
                loss_history.append(loss_components)




                # #step_in_current_epoch = gs - (epoch * g.num_batch)
                # #print (step_in_current_epoch)
                # if step_in_current_epoch in validate_steps:
                #     fractional_epoch = epoch + validate_steps[step_in_current_epoch]
                #     #info('fractional_epoch: {0:0.3f}'.format(fractional_epoch))

                #     loss_history = np.array(loss_history)
                #     train_loss_mean_std = np.concatenate([loss_history.mean(axis=0), loss_history.std(axis=0)])
                #     loss_history = []

                #     train_loss_mean_std = ' '.join(['{:0.3f}'.format(score) for score in train_loss_mean_std])
                #     info('train epoch {0:0.3f}: {1}'.format(fractional_epoch, train_loss_mean_std))

                #     current_score = compute_validation(hp, num, fractional_epoch, validation_inputs, synth_graph, sess, speaker_codes, valid_filenames, validation_reference)
                #     info('validation epoch {0:0.3f}: {1:0.3f}'.format(fractional_epoch, current_score))
                #     #info('Current score: {}; best so far: {}'.format(current_score, best_score))
                    
                #     if current_score < best_score:
                #         checks_since_best = 0
                #         best_score = current_score
                #         info('New best score at epoch {0:0.3f}'.format(fractional_epoch))
                #         ### For now, just save at regular intervals with save_every_n_epochs
                #         #sv.saver.save(sess, logdir + '/model_epoch_{0:0.3f}'.format(fractional_epoch))
                #     else:
                #         checks_since_best += 1

                #     if checks_since_best > patience:
                #         info('patience ({0} epochs) exceeded: end training at epoch {1:0.3}'.format(hp.patience_epochs, fractional_epoch)); return 
                #     # if num==1:
                #     #     # plot alignment
                #     #     alignments = sess.run(g.alignments)
                #     #     plot_alignment(hp, alignments[0], str(gs // save_every).zfill(3) + "k", logdir)


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
                    
            ### Save end of each epoch:       
            stem = logdir + '/model_epoch_{0}'.format(epoch)
            sv.saver.save(sess, stem)

            ### Check if we should archive:
            if hp.save_every_n_epochs:
                if epoch % hp.save_every_n_epochs == 0:
                    info('Archive model %s'%(stem))
                    for fname in glob.glob(stem + '*'):
                        shutil.copy(fname, logdir + '/archive/')


            # if hp.save_every_n_epochs:
            #     if epoch % hp.save_every_n_epochs == 0:
            #         stem = logdir + '/model_epoch_{0:0.3f}'.format(fractional_epoch)
            #         info('Archive model %s'%(stem))
            #         if os.path.isfile(stem + '.index'):
            #             for fname in glob.glob(stem + '*'):
            #                 shutil.copy(fname, logdir + '/archive/')
            #         else:
            #             sv.saver.save(sess, stem)
            #             for fname in glob.glob(stem + '*'):
            #                 shutil.move(fname, logdir + '/archive/')

            epoch += 1
            if epoch > hp.max_epochs: 
                info('Max epochs ({}) reached: end training'.format(hp.max_epochs)); return

    print("Done")


if __name__=="__main__":

    main_work()



