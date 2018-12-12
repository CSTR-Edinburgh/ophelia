#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - March 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser

import soundfile as sf

from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm


HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
sv56 = HERE + '/../tool/bin/sv56demo'

if not os.path.isfile(sv56):

    ## Check required executables are available:

    from distutils.spawn import find_executable

    required_executables = ['sv56demo']

    for executable in required_executables:
        if not find_executable(executable):
            sys.exit('%s command line tool must be on system path '%(executable))
        
    sv56 = 'sv56demo'


def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='indir', required=True)    
    a.add_argument('-o', dest='outdir', required=True, \
                    help= "Put output here: make it if it doesn't exist")
    a.add_argument('-pattern', default='', \
                    help= "If given, only normalise files whose base contains this substring")
    a.add_argument('-ncores', default=1, type=int)
    opts = a.parse_args()
    
    # ===============================================
    
    for direc in [opts.outdir]:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    flist = sorted(glob.glob(opts.indir + '/*.wav'))
    
    executor = ProcessPoolExecutor(max_workers=opts.ncores)
    futures = []
    for wave_file in flist:
        futures.append(executor.submit(
            partial(process, wave_file, opts.outdir, pattern=opts.pattern)))
    return [future.result() for future in tqdm(futures)]






def process(wavefile, outdir, pattern=''):
    _, base = os.path.split(wavefile)
    
    if pattern:
        if pattern not in base:
            return

    # print base

    raw_in = os.path.join(outdir, base.replace('.wav','.raw'))
    raw_out = os.path.join(outdir, base.replace('.wav','_norm.raw'))
    logfile = os.path.join(outdir, base.replace('.wav','.log'))
    wav_out = os.path.join(outdir, base)
    
    data, samplerate = sf.read(wavefile, dtype='int16')
    sf.write(raw_in, data, samplerate, subtype='PCM_16')
    os.system('%s -log %s -q -lev -26.0 -sf %s %s %s'%(sv56, logfile, samplerate, raw_in, raw_out))
    norm_data, samplerate = sf.read(raw_out, dtype='int16', samplerate=samplerate, channels=1, subtype='PCM_16')
    sf.write(wav_out, norm_data, samplerate)

    os.system('rm %s %s'%(raw_in, raw_out))




if __name__=="__main__":

    main_work()

