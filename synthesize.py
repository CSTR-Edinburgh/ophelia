# -*- coding: utf-8 -*-
# /usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
from graphs import Graph
from utils import *
from data_load import load_data
from scipy.io.wavfile import write
from tqdm import tqdm

## to import configs
# HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append( HERE + '/config/' )
# import importlib
import imp


from argparse import ArgumentParser


import timeit 

def start_clock(comment):
    print ('%s... '%(comment)),
    return (timeit.default_timer(), comment)

def stop_clock((start_time, comment), width=40):
    padding = (width - len(comment)) * ' '
    print ('%s--> took %.2f seconds' % (padding, (timeit.default_timer() - start_time)) )

def denorm(data, stats, type):
    if type=='minmax':
        mini = stats[0,:].reshape(1,-1)
        maxi = stats[1,:].reshape(1,-1)  
        X = data   
        # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (maxi - mini) + mini   
        return X_scaled        
    elif type=='meanvar':
        mean = stats[0,:].reshape(1,-1)
        std = stats[1,:].reshape(1,-1)
        data = (data * std) + mean
    else:
        sys.exit('Unknown normtype: %s'%(type))
    return data

def synthesize(hp, speaker_id=''):
    assert hp.vocoder=='griffin_lim', 'Other vocoders than griffin_lim not yet supported'

    # Load data
    (fpaths, L) = load_data(hp, mode="synthesis")
    L = L[:3,:]

    if speaker_id:
        speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))
        speaker_ix = speaker2ix[speaker_id]
        speaker_data = np.zeros_like(L)
        
        for i in range(len(L)):
            length = (np.where(L[i,:]==0)[0][0])
            speaker_data[i,:length] = speaker_ix

    # Load graph
    g = Graph(hp,  mode="synthesize"); print("Graph 1 loaded")
    #g = Graph(hp, num=2, mode="synthesize"); print("Graph 2 loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        # Restore parameters
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'Text2Mel')
        saver1 = tf.train.Saver(var_list=var_list)
        savepath = hp.logdir + "-1"
        print ([savepath])
        saver1.restore(sess, tf.train.latest_checkpoint(savepath))
        print("Text2Mel Restored!")

        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') + \
                   tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'gs')
        saver2 = tf.train.Saver(var_list=var_list)
        savepath = hp.logdir + "-2"        
        saver2.restore(sess, tf.train.latest_checkpoint(savepath))
        print("SSRN Restored!")

        # Feed Forward
        ## mel
        Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
        prev_max_attentions = np.zeros((len(L),), np.int32)

        t = start_clock('gen')
        for j in tqdm(range(hp.max_T)):
            if hp.multispeaker:
                _Y, _max_attentions, _alignments, = \
                    sess.run([ g.Y, g.max_attentions, g.alignments],
                             {g.L: L,
                              g.mels: Y,
                              g.speakers: speaker_data,
                              g.prev_max_attentions: prev_max_attentions}) ##
            else:
                _Y, _max_attentions, _alignments, = \
                    sess.run([ g.Y, g.max_attentions, g.alignments],
                             {g.L: L,
                              g.mels: Y,
                              g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop
            Y[:, j, :] = _Y[:, j, :]
            prev_max_attentions = _max_attentions[:, j]
        stop_clock(t)


        print ('get mag...')
        # Get magnitude
        Z = sess.run(g.Z, {g.Y: Y})

        # Generate wav files
        if not os.path.exists(hp.sampledir): os.makedirs(hp.sampledir)
        for i, mag in enumerate(Z):
            print("Working on file", i+1)
            if hp.vocoder=='magphase_compressed':
                mag = denorm(mag, s, hp.normtype)
                streams = split_streams(mag, ['mag', 'lf0', 'vuv', 'real', 'imag'], [60,1,1,45,45])
                wav = magphase_synth_from_compressed(streams, samplerate=hp.sr)
            elif hp.vocoder=='griffin_lim':
                wav = spectrogram2wav(hp, mag)
            else:
                sys.exit('evlsdvlsvsvbsfbv')
            write(hp.sampledir + "/{}.wav".format(i+1), hp.sr, wav)





def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-speaker', default='', type=str)
    opts = a.parse_args()
    
    # ===============================================
    
    config = os.path.abspath(opts.config)
    assert os.path.isfile(config)

    conf_mod = imp.load_source('config', config)
    hp = conf_mod.Hyperparams()

    if hp.multispeaker:
        assert opts.speaker
        assert opts.speaker in hp.speaker_list

    synthesize(hp, speaker_id=opts.speaker)


if __name__=="__main__":

    main_work()
