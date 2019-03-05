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

def copy_synth_GL(hp, outdir):

    safe_makedir(outdir)

    dataset = load_data(hp, mode="synthesis") 
    fnames, texts = dataset['fpaths'], dataset['texts']
    bases = [basename(fname) for fname in fnames]
    
    for base in bases:
        print("Working on file %s"%(base))
        mag = np.load(os.path.join(hp.full_audio_dir, base + '.npy'))
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
    copy_synth_GL(hp, opts.outdir)

if __name__=="__main__":

    main_work()
