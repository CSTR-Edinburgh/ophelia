# -*- coding: utf-8 -*-
#! /usr/bin/env python2


from __future__ import print_function

import os
import sys
import glob
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import numpy as np



from sklearn.preprocessing import StandardScaler, MinMaxScaler

from tqdm import tqdm

from libutil import safe_makedir, basename
# from configuration import load_config
# from utils import load_spectrograms

from interpolate_unvoiced import interpolate_through_unvoiced

def get_speech(infile, dim):
    #print (infile)
    f = open(infile, 'rb')
    data = np.fromfile(f, dtype=np.float32)
    f.close()
    assert data.size % float(dim) == 0.0,'specified dimension %s not compatible with data'%(dim)
    data = data.reshape((-1, dim))
    return data



def load_sentence(fpath, worlddir='', outdir=''):
    assert worlddir and outdir, ()
    mel = np.load(fpath)

    #print (mel.shape)
         

    base = basename(fpath)
    streams = []
    for (stream, dim) in [('lf0', 1),('mgc', 60),('bap', 1)]:
        fname = '%s/%s/%s.%s'%(worlddir, stream, base, stream)
        speech  = get_speech(fname, dim)
        #print (fname)
        #print (speech.shape)
        if stream=='lf0':
            speech, vuv = interpolate_through_unvoiced(speech)
            streams.extend([speech, vuv])
        else:
            streams.append(speech)
    composed = np.hstack(streams)

    target_frames, _ = mel.shape
    actual_frames, _ = composed.shape

    #print (target_frames, actual_frames)
    diff = target_frames - actual_frames
    if diff < 0:
        sys.exit('world features too short')
    elif diff > 0:
        composed = np.pad(composed,((0,diff),(0,0)),mode='constant')

    return composed


def process(fpath, worlddir='', outdir='', scaler=''):
    assert scaler
    speech = load_sentence(fpath, worlddir=worlddir, outdir=outdir)
    norm_speech = standardise_acoustics(speech, scaler)
    np.save('%s/full_world/%s'%(outdir, basename(fpath)), norm_speech)
    np.save('%s/coarse_world/%s'%(outdir, basename(fpath)), norm_speech[::4, :])


def update_normalisation_stats(acoustic_data, scaler):
    '''
    Partially update stats in external scaler_dict on sentence's acoustic data.
    Modify the scaler_dict in-place.
    '''
    scaler.partial_fit(acoustic_data)
    return scaler
    

def standardise_acoustics(acoustic_data, scaler):
    '''
    Use external scaler_dict to standardise this sentence's acoustic streams
    '''

    norm_acoustic_data = scaler.transform(acoustic_data)
    return norm_acoustic_data



def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-meldir', required=True, type=str, help='existing directory with mels - features are padding to match length of these ')
    a.add_argument('-worlddir', required=True, type=str, help='existing directory containing world features')    
    a.add_argument('-outdir', required=True, type=str)    
    
    a.add_argument('-testpatt', required=False, type=str, default='')  


    a.add_argument('-ncores', default=1, type=int, help='Number of cores for parallel processing')    
    opts = a.parse_args()
    
    # ===============================================

    # hp = load_config(opts.config)

    fpaths = sorted(glob.glob(opts.meldir + '/*.npy')) # [:10]


    normkind='meanvar'

    if normkind=='minmax':
        scaler =  MinMaxScaler()        
    elif normkind=='meanvar':
        scaler = StandardScaler()
    else:
        sys.exit('aedvsv')


    
    if opts.testpatt:
        train_fpaths = [p for p in fpaths if opts.testpatt not in basename(p)]
    else:
        train_fpaths = fpaths

    for fpath in tqdm(train_fpaths, desc='First pass to get norm stats'):

        data = load_sentence(fpath, worlddir=opts.worlddir, outdir=opts.outdir)
        scaler = update_normalisation_stats(data, scaler)


    safe_makedir(opts.outdir)
    safe_makedir(opts.outdir + '/full_world/')
    safe_makedir(opts.outdir + '/coarse_world/')

    if 0:
        process(fpaths[0], worlddir=opts.worlddir, outdir=opts.outdir, scaler=scaler)
        sys.exit('aedvsfv')

    executor = ProcessPoolExecutor(max_workers=opts.ncores)
    futures = []
    for fpath in fpaths:
        futures.append(executor.submit(
            process, fpath, worlddir=opts.worlddir, outdir=opts.outdir, scaler=scaler))

    proc_list =  [future.result() for future in tqdm(futures, desc='Second pass (parallel) to do normalisation')]

    if normkind=='minmax':
        mini = scaler.data_min_  ## TODO: per speaker...
        maxi = scaler.data_max_
        stats = np.vstack([mini, maxi])
    elif normkind=='meanvar':
        mean = scaler.mean_  ## TODO: per speaker...
        std = scaler.scale_
        stats = np.vstack([mean, std])
    else:
        sys.exit('aedvsv2')
    np.save(opts.outdir + '/norm_stats', stats)





    # safe_makedir(opts.outdir)
         
    # executor = ProcessPoolExecutor(max_workers=opts.ncores)    
    # futures = []
    # for fpath in fpaths:
    #     futures.append(executor.submit(
    #         proc, fpath, worlddir=opts.worlddir, outdir=opts.outdir))
    # proc_list = [future.result() for future in tqdm.tqdm(futures)]


if __name__=="__main__":

    main_work()
