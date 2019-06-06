#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2019 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk


import sys
import os
import glob
from argparse import ArgumentParser

from libutil import get_speech, basename, safe_makedir
from scipy.signal import argrelextrema
import numpy as np
import matplotlib as mpl
mpl.use('PDF')
import pylab as pl 

from scipy import interpolate


# def merlin_state_label_to_phone(labfile):
#     labels = np.loadtxt(labfile, dtype=str, comments=None) ## default comments='#' breaks
#     starts = labels[:,0].astype(int)[::5].reshape(-1,1)
#     ends = labels[:,1].astype(int)[4::5].reshape(-1,1)
#     fc = labels[:,2][::5]
#     fc = np.array([line.replace('[2]','') for line in fc]).reshape(-1,1)
#     phone_label = np.hstack([starts, ends, fc])
#     return phone_label


def minmax_norm(X, data_min, data_max):
    data_range = data_max - data_min
    data_range[data_range<=0.0] = 1.0
    maxi, mini = 0.01, 0.99   # ## merlin's default desired range
    X_std = (X - data_min) / data_range
    X_scaled = X_std * (maxi - mini) + mini    
    return X_scaled


def process_merlin_positions(bin_label_fname, audio_dir, phonedim=416, subphonedim=9, \
                    inrate=5.0, outrate=12.5):

    audio_fname = os.path.join(audio_dir, basename(bin_label_fname) + '.npy')
    assert os.path.isfile(audio_fname), 'No audio file for %s '%(basename(bin_label_fname))
    audio = np.load(audio_fname)

    labfrombin = get_speech(bin_label_fname, phonedim+subphonedim)
    
    positions = labfrombin[:,-subphonedim:] 

    nframes, dim = positions.shape
    assert dim==9

    new_nframes, _ = audio.shape

    old_x = np.linspace((inrate/2.0), nframes*inrate, nframes, endpoint=False)  ## place points at frame centres
    
    f = interpolate.interp1d(old_x, positions, axis=0, kind='nearest', bounds_error=False, fill_value='extrapolate') ## nearest to avoid weird averaging effects near segment boundaries

    new_x = np.linspace((outrate/2.0), new_nframes*outrate, new_nframes, endpoint=False)
    new_positions = f(new_x)  
    
    return new_positions



def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()

    a.add_argument('-b', dest='binlabdir', required=True)   
    a.add_argument('-f', dest='audio_dir', required=True)    
    a.add_argument('-n', dest='norm_info_fname', required=True)  
    a.add_argument('-o', dest='outdir', required=True) 
    a.add_argument('-binext', dest='binext', required=False, default='lab')    

    a.add_argument('-ir', dest='inrate', type=float, default=5.0) 
    a.add_argument('-or', dest='outrate', type=float, default=12.5)   

    opts = a.parse_args()
    
    # ===============================================

    safe_makedir(opts.outdir)

    norm_info = get_speech(opts.norm_info_fname, 425)[:,-9:]
    data_min = norm_info[0,:]
    data_max = norm_info[1,:]
    data_range = data_max - data_min

    audio_files = set([basename(f) for f in glob.glob(opts.audio_dir + '/*.npy')])
    binary_label_files = sorted(glob.glob(opts.binlabdir + '/*.' + opts.binext) )
    
    for binlab in binary_label_files:
        base = basename(binlab)
        if base not in audio_files:
            continue
        print base
        positions = process_merlin_positions(binlab, opts.audio_dir, inrate=opts.inrate, outrate=opts.outrate)
        norm_positions = minmax_norm(positions, data_min, data_max)

        np.save(opts.outdir + '/' + base, norm_positions)


if __name__=="__main__":
    main_work()

