# -*- coding: utf-8 -*-
# /usr/bin/python2

from __future__ import print_function

import os
import glob

import numpy as np
from utils import spectrogram2wav
from data_load import load_data
import soundfile
from tqdm import tqdm
from configuration import load_config

from argparse import ArgumentParser

from libutil import basename, safe_makedir
from synthesize import synth_mel2mag, list2batch, restore_latest_model_parameters
from architectures import SSRNGraph
import tensorflow as tf

def copy_synth_SSRN_GL(hp, outdir):

    safe_makedir(outdir)

    dataset = load_data(hp, mode="synthesis") 
    fnames, texts = dataset['fpaths'], dataset['texts']
    bases = [basename(fname) for fname in fnames]
    mels = [np.load(os.path.join(hp.coarse_audio_dir, base + '.npy')) for base in bases]
    lengths = [a.shape[0] for a in mels]
    mels = list2batch(mels, 0)

    g = SSRNGraph(hp, mode="synthesize"); print("Graph (ssrn) loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        ssrn_epoch = restore_latest_model_parameters(sess, hp, 'ssrn')

        print('Run SSRN...')
        Z = synth_mel2mag(hp, mels, g, sess)

        for i, mag in enumerate(Z):
            print("Working on %s"%(bases[i]))
            mag = mag[:lengths[i]*hp.r,:]  ### trim to generated length             
            wav = spectrogram2wav(hp, mag)
            soundfile.write(outdir + "/%s.wav"%(base), wav, hp.sr)

  


def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-o', dest='outdir', required=True, type=str)    
    opts = a.parse_args()
    
    # ===============================================
    
    hp = load_config(opts.config)
    copy_synth_SSRN_GL(hp, opts.outdir)

if __name__=="__main__":

    main_work()
