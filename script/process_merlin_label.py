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

def merlin_state_label_to_phone(labfile):
    labels = np.loadtxt(labfile, dtype=str, comments=None) ## default comments='#' breaks
    starts = labels[:,0].astype(int)[::5].reshape(-1,1)
    ends = labels[:,1].astype(int)[4::5].reshape(-1,1)
    fc = labels[:,2][::5]
    fc = np.array([line.replace('[2]','') for line in fc]).reshape(-1,1)
    phone_label = np.hstack([starts, ends, fc])
    return phone_label


def minmax_norm(X, data_min, data_max):
    data_range = data_max - data_min
    data_range[data_range<=0.0] = 1.0
    maxi, mini = 0.01, 0.99   # ## merlin's default desired range
    X_std = (X - data_min) / data_range
    X_scaled = X_std * (maxi - mini) + mini    
    return X_scaled


def process_merlin_label(bin_label_fname, text_lab_dir, phonedim=416, subphonedim=9):

    text_label = os.path.join(text_lab_dir, basename(bin_label_fname) + '.lab')
    assert os.path.isfile(text_label), 'No text file for %s '%(basename(bin_label_fname))
    
    labfrombin = get_speech(bin_label_fname, phonedim+subphonedim)
    
    ## fraction through phone (forwards)
    fraction_through_phone_forwards = labfrombin[:,-1]

    ## This is a suprisingly noisy signal which never seems to start at 0.0! Find minima:-
    (minima, ) = argrelextrema(fraction_through_phone_forwards, np.less)

    ## first frame is always a start: 
    minima = np.insert(minima, 0, 0)  

    ## check size against text file:
    labfromtext = merlin_state_label_to_phone(text_label)
    assert labfromtext.shape[0] == minima.shape[0]

    lab = labfrombin[minima,:-subphonedim] ## discard frame level feats, and take first frame of each phone

    return lab



def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()

    a.add_argument('-b', dest='binlabdir', required=True)   
    a.add_argument('-t', dest='text_lab_dir', required=True)    
    a.add_argument('-n', dest='norm_info_fname', required=True)  
    a.add_argument('-o', dest='outdir', required=True) 
    a.add_argument('-binext', dest='binext', required=False, default='lab')    

    opts = a.parse_args()
    
    # ===============================================

    safe_makedir(opts.outdir)

    norm_info = get_speech(opts.norm_info_fname, 425)[:,:-9]
    data_min = norm_info[0,:]
    data_max = norm_info[1,:]
    data_range = data_max - data_min

    text_label_files = set([basename(f) for f in glob.glob(opts.text_lab_dir + '/*.lab')])
    binary_label_files = sorted(glob.glob(opts.binlabdir + '/*.' + opts.binext) )
    print binary_label_files
    for binlab in binary_label_files:
        base = basename(binlab)
        if base not in text_label_files:
            continue
        print base
        lab = process_merlin_label(binlab, opts.text_lab_dir)
        norm_lab = minmax_norm(lab, data_min, data_max)

        if 0: ## piano roll style plot:
            pl.imshow(norm_lab, interpolation='nearest')
            pl.gray()
            pl.savefig('/afs/inf.ed.ac.uk/user/o/owatts/temp/fig.pdf')
            sys.exit('abckdubv')

        np.save(opts.outdir + '/' + base, norm_lab)


if __name__=="__main__":
    main_work()

