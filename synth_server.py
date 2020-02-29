# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import os

import numpy as np
import tensorflow as tf
# from graphs import Graph
from utils import *
from data_load import load_vocab, phones_normalize, text_normalize
from scipy.io.wavfile import write
from tqdm import tqdm

## to import configs
HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
sys.path.append( HERE + '/script/' )

import sys
import signal
import imp
import codecs
import glob
import timeit 
import time
import os
import stat
from argparse import ArgumentParser

import datetime
import time

from configuration import load_config
from synthesize import restore_archived_model_parameters, restore_latest_model_parameters, \
                        get_text_lengths, encode_text, synth_codedtext2mel, synth_mel2mag, \
                        synth_wave
from expand_digits import norm_hausa
from festival.flite import get_flite_phonetisation

from architectures import Text2MelGraph, SSRNGraph

def file_age_in_seconds(pathname):
    return time.time() - os.stat(pathname)[stat.ST_MTIME]

def start_clock(comment):
    print ('%s... '%(comment)),
    return (timeit.default_timer(), comment)

def stop_clock((start_time, comment), width=40):
    padding = (width - len(comment)) * ' '
    print ('%s--> took %.2f seconds' % (padding, (timeit.default_timer() - start_time)) )

def signal_handler(signal, frame):
    print("\nStopping the server")
    sys.exit(0)





def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-dir', dest='synthdir', required=True, type=str)
    a.add_argument('-limit', default=0, type=int)   
    a.add_argument('-controllable', action='store_true', default=False) 
    opts = a.parse_args()
    
    # ===============================================
    
    hp = load_config(opts.config)

    hp.language = 'en_cmulex'



    if hp.language=='en_cmulex':
        s = CMULexSynthesiser(hp, controllable=opts.controllable)
    elif hp.language=='hausa':
        s = HausaSynthesiser(hp, controllable=opts.controllable)
    else:
        s = Synthesiser(hp, controllable=opts.controllable)

    if opts.limit:
        s.hardlimit = opts.limit

    if 0: ## for debugging basic system
        os.system('echo "The fish twisted and turned." > /tmp/test.txt')
        s.synthesise('/tmp/test.txt', '/afs/inf.ed.ac.uk/group/cstr/projects/scar/SCRIPT/temp/stest/utt01.wav')
        sys.exit('qlebfcwivbrev88888')


    if 0: ## for debugging extended system with control vectors 
        os.system('echo "The fish twisted and turned." > /tmp/test.txt')
        s.synthesise('/tmp/test.txt', '/afs/inf.ed.ac.uk/group/cstr/projects/scar/SCRIPT/temp/stest/utt04.wav', control_vector=[-0.8, -0.5]) # [0.8, -0.5])
        sys.exit('vksfj vksfe298f8')


    # https://stackoverflow.com/questions/18994912/ending-an-infinite-while-loop
    signal.signal(signal.SIGINT, signal_handler)

    while True:
        s.remove_old_wave_files(opts.synthdir)
        s.check_for_new_files_and_synth(opts.synthdir)
        time.sleep(0.5)
        print( datetime.datetime.now() )
        print('Listening for new .txt files with no .wav associated at %s'%(opts.synthdir))


if __name__=="__main__":

    main_work()