# -*- coding: utf-8 -*-
#/usr/bin/env python2

from __future__ import print_function

from utils import get_attention_guide
import os
# import sys
from data_load import load_data
import numpy as np
import tqdm
# import glob

from concurrent.futures import ProcessPoolExecutor

# HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
# sys.path.append( HERE + '/config/' )
# import importlib
import imp

from argparse import ArgumentParser

from libutil import basename, save_floats_as_8bit

def proc(fpath, text_length, hp):
    
    base = basename(fpath)
    melfile = hp.coarse_audio_dir + os.path.sep + base + '.npy'
    attfile = hp.attention_guide_dir + os.path.sep + base # without '.npy'
    if not os.path.isfile(melfile):
        print('file %s not found'%(melfile))
        return
    speech_length = np.load(melfile).shape[0]
    att = get_attention_guide(text_length, speech_length, g=hp.g)
    #print (text_length, speech_length)
    save_floats_as_8bit(att, attfile)
    #print ('Wrote %s'%(attfile))






def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-ncores', default=1, type=int, help='Number of cores for parallel processing')    
    opts = a.parse_args()
    
    # ===============================================

    config = os.path.abspath(opts.config)
    assert os.path.isfile(config)

    conf_mod = imp.load_source('config', config)
    hp = conf_mod.Hyperparams()

    if not hp.attention_guide_dir:
        sys.exit('attention_guide_dir not specified in config')

    [fpaths, text_lengths] = load_data(hp)[:2]
    #fpaths = sorted(glob.glob(hp.waveforms + '/*.wav'))
    # print(zip(fpaths, text_lengths))
    assert os.path.exists(hp.coarse_audio_dir)
    if not os.path.exists(hp.attention_guide_dir): os.makedirs(hp.attention_guide_dir)

    executor = ProcessPoolExecutor(max_workers=opts.ncores)    
    futures = []
    for (fpath, text_length) in zip(fpaths, text_lengths):
    #    proc(fpath, text_length, hp)
         futures.append(executor.submit(proc, fpath, text_length, hp)) 
    proc_list = [future.result() for future in tqdm.tqdm(futures)]



if __name__=="__main__":

    main_work()
