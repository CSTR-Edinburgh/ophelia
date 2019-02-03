# -*- coding: utf-8 -*-
#! /usr/bin/env python2
'''
Based on code by kyubyong park at https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os
import sys
import glob
from argparse import ArgumentParser
from concurrent.futures import ProcessPoolExecutor
import numpy as np

import tqdm

from data_load import load_data
from libutil import safe_makedir, load_config
from utils import load_spectrograms

def proc(fpath, hp):
    
    if not os.path.isfile(fpath):
        return
        
    fname, mel, mag, full_mel = load_spectrograms(hp, fpath)
    np.save("{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")), mel)
    np.save("{}/{}".format(hp.full_audio_dir, fname.replace("wav", "npy")), mag)
    np.save("{}/{}".format(hp.full_mel_dir, fname.replace("wav", "npy")), full_mel)


def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-ncores', default=1, type=int, help='Number of cores for parallel processing')    
    opts = a.parse_args()
    
    # ===============================================

    hp = load_config(opts.config)

    fpaths = sorted(glob.glob(hp.waveforms + '/*.wav'))

    safe_makedir(hp.coarse_audio_dir)
    safe_makedir(hp.full_audio_dir)
    safe_makedir(hp.full_mel_dir)
           
    executor = ProcessPoolExecutor(max_workers=opts.ncores)    
    futures = []
    for fpath in fpaths:
        futures.append(executor.submit(
            proc, fpath, hp))
    proc_list = [future.result() for future in tqdm.tqdm(futures)]


if __name__=="__main__":

    main_work()
