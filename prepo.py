# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

from utils import load_spectrograms
import os
import sys
from data_load import load_data
import numpy as np
import tqdm

from concurrent.futures import ProcessPoolExecutor

HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append( HERE + '/config/' )
import importlib



def proc(fpath):
    
    if hp.extract_full_mel:
        fname, mel, mag, full_mel = load_spectrograms(hp, fpath)
    else:
        fname, mel, mag = load_spectrograms(hp, fpath)
    np.save("{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy")), mel)
    np.save("{}/{}".format(hp.full_audio_dir, fname.replace("wav", "npy")), mag)
    if hp.extract_full_mel:
        np.save("{}/{}".format(hp.full_mel_audio_dir, fname.replace("wav", "npy")), full_mel)





def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-ncores', default=1, type=int, help='Number of cores for parallel processing')    
    opts = a.parse_args()
    
    # ===============================================
    num = opts.num
    config = opts.config

    conf_mod = importlib.import_module(config)
    hp = conf_mod.Hyperparams()

    # Load data
    fpaths, _, _ = load_data(hp) # list

    if not os.path.exists(hp.coarse_audio_dir): os.makedirs(hp.coarse_audio_dir)
    if not os.path.exists(hp.full_audio_dir): os.makedirs(hp.full_audio_dir)
    if hp.extract_full_mel:
        if not os.path.exists(hp.full_mel_audio_dir): os.makedirs(hp.full_mel_audio_dir)
           
    executor = ProcessPoolExecutor(max_workers=opts.ncores)    
    futures = []
    for fpath in fpaths:
        futures.append(executor.submit(
            proc, fpath))
    proc_list = [future.result() for future in tqdm.tqdm(futures)]


if __name__=="__main__":

    main_work()
