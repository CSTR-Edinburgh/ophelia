#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
from argparse import ArgumentParser

import imp

import numpy as np

from utils import spectrogram2wav
from scipy.io.wavfile import write

import tqdm
from concurrent.futures import ProcessPoolExecutor

import tensorflow as tf
from architectures import SSRNGraph
from synthesize import make_mel_batch, split_batch, synth_mel2mag
from configuration import load_config


def synth_wave(hp, magfile):
    mag = np.load(magfile)
    #print ('mag shape %s'%(str(mag.shape)))
    wav = spectrogram2wav(hp, mag)
    outfile = magfile.replace('.mag.npy', '.wav')
    outfile = outfile.replace('.npy', '.wav')
    #print magfile
    #print outfile
    #print 
    write(outfile, hp.sr, wav)

def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-ncores', type=int, default=1)
    opts = a.parse_args()
    
    # ===============================================

    hp = load_config(opts.config)
    
    ### 1) convert saved coarse mels to mags with latest-trained SSRN
    print('mel2mag: restore last saved SSRN')
    g = SSRNGraph(hp,  mode="synthesize")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        
        var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'SSRN') 
        saver2 = tf.train.Saver(var_list=var_list)
        savepath = hp.logdir + "-ssrn"        
        latest_checkpoint = tf.train.latest_checkpoint(savepath)
        ssrn_epoch = latest_checkpoint.strip('/ ').split('/')[-1].replace('model_epoch_', '')
        saver2.restore(sess, latest_checkpoint)
        print("SSRN Restored from latest epoch %s"%(ssrn_epoch))

        filelist = glob.glob(hp.logdir + '-t2m/validation_epoch_*/*.npy')
        filelist = [fname for fname in filelist if not fname.endswith('.mag.npy')]
        batch, lengths = make_mel_batch(hp, filelist, oracle=False)
        Z = synth_mel2mag(hp, batch, g, sess, batchsize=32)
        print ('synthesised mags, now splitting batch:')
        maglist = split_batch(Z, lengths)
        for (infname, outdata) in tqdm.tqdm(zip(filelist, maglist)):
            np.save(infname.replace('.npy','.mag.npy'), outdata)



    ### 2) GL in parallel for both t2m and ssrn validation set 
    print('GL for SSRN validation')
    filelist = glob.glob(hp.logdir + '-t2m/validation_epoch_*/*.mag.npy') + \
               glob.glob(hp.logdir + '-ssrn/validation_epoch_*/*.npy')

    if opts.ncores==1:
        for fname in tqdm.tqdm(filelist):
            synth_wave(hp, fname)
    else:
        executor = ProcessPoolExecutor(max_workers=opts.ncores)    
        futures = []
        for fpath in filelist:
            futures.append(executor.submit(synth_wave, hp, fpath))
        proc_list = [future.result() for future in tqdm.tqdm(futures)]



if __name__=="__main__":

    main_work()
