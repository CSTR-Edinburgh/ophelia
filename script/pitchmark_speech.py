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
import subprocess

import  numpy as np
#from generate import read_est_file

## Check required executables are available:

from distutils.spawn import find_executable

# required_executables = ['reaper']

# for executable in required_executables:
#     if not find_executable(executable):
#         sys.exit('%s command line tool must be on system path '%(executable))

# _reaper_bin = 'reaper'    

HERE = os.path.realpath(os.path.abspath(os.path.dirname(__file__)))
bindir = HERE + '/../tool/bin/'

_reaper_bin = bindir + '/reaper'    




if not (os.path.isfile(_reaper_bin) and os.access(_reaper_bin, os.X_OK)):
    executables = ['reaper']
    for executable in executables:
        if not find_executable(executable):
             sys.exit('%s command line tool must be on system path '%(executable))
    _reaper_bin = 'reaper'  



## TODO : copied from generate
def read_est_file(est_file):

    with open(est_file) as fid:
        header_size = 1 # init
        for line in fid:
            if line == 'EST_Header_End\n':
                break
            header_size += 1
        ## now check there is at least 1 line beyond the header:
        status_ok = False
        for (i,line) in enumerate(fid):
            if i > header_size:
                status_ok = True
    if not status_ok:
        return np.array([])

    # Read text: TODO: improve skiprows
    data = np.loadtxt(est_file, skiprows=header_size)
    data = np.atleast_2d(data)
    return data


def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='indir', required=True)    
    a.add_argument('-o', dest='outdir', required=True, \
                    help= "Put output here: make it if it doesn't exist") 
    a.add_argument('-f', dest='fzdir', required=True, \
                    help= "Put F0 traces here: make it if it doesn't exist")  
    a.add_argument('-framerate', required=False, default=0.005, type=float, help='rate in seconds for F0 track frames')
    a.add_argument('-pattern', default='', \
                    help= "If given, only normalise files whose base contains this substring")
    a.add_argument('-ncores', default=1, type=int)
    #a.add_argument('-waveformat', default=False, action='store_true', help='call sox to format data (16 bit ).')    
    
    a.add_argument('-twopass', default=False, action='store_true', help='Run initially on a subset of data to guess sensible limits, then run again. Assumes all data is from same speaker.')    
    opts = a.parse_args()
    
    # ===============================================
    
    for direc in [opts.outdir, opts.fzdir]:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    flist = sorted(glob.glob(opts.indir + '/*.wav'))
    
    if opts.twopass:
        temp_outdir = opts.outdir.rstrip('/ ') + '_temp'
        temp_fzdir = opts.fzdir.rstrip('/ ') + '_temp'
        for direc in [temp_outdir, temp_fzdir]:
            if not os.path.isdir(direc):
                os.makedirs(direc)

        ## first pass with no min and max:
        executor = ProcessPoolExecutor(max_workers=opts.ncores)
        futures = []

        ## first pass on small subset of data only
        if flist > 100:
            short_flist = flist[:100]
        else:
            short_flist = flist
        for wave_file in short_flist:
            futures.append(executor.submit(
                partial(process, wave_file, temp_outdir, temp_fzdir, pattern=opts.pattern, framerate=opts.framerate)))
        proc_list =  [future.result() for future in tqdm(futures)]

        ## compute thresholds:
        fzero_flist = glob.glob(temp_fzdir + '/*.f0')
        all_voiced = []
        for fname in tqdm(fzero_flist):
            d = read_est_file(fname)
            fz = d[:,2]
            voiced = fz[fz > 0.0]
            all_voiced.append(voiced)

        all_voiced = np.hstack(all_voiced)
            
        # De Looze and Rauzy (2009):
        min_f0 = int((np.percentile(all_voiced, 35) * 0.72) - 10.0)
        max_f0 = int((np.percentile(all_voiced, 65) * 1.9) + 10.0)

    else:
        min_f0 = 50        
        max_f0 = 400

    print 'Extract with range %s %s'%(min_f0, max_f0)
    executor = ProcessPoolExecutor(max_workers=opts.ncores)
    futures = []
    for wave_file in flist:
        futures.append(executor.submit(
            partial(process, wave_file, opts.outdir, opts.fzdir, pattern=opts.pattern, min_f0=min_f0, max_f0=max_f0, framerate=opts.framerate)))
    return [future.result() for future in tqdm(futures)]






def process(wavefile, outdir, fzdir, pattern='', max_f0=400, min_f0=50, framerate=0.005):
    _, base = os.path.split(wavefile)
    base = base.replace('.wav', '')
    
    if pattern:
        if pattern not in base:
            return

    out_est_file = os.path.join(outdir, base + '.pm')
    out_fz_file = os.path.join(fzdir, base + '.f0')

    ##### preprocess waveform (artci 32k only)
    # in_wav_file = os.path.join(fzdir, base + '_tmp.wav')  ### !!!!!
    # cmd = 'sox %s -r 16000 %s '%(wavefile, in_wav_file)
    # subprocess.call(cmd, shell=True)
    # cmd =  _reaper_bin + " -s -e %s -x %s -m %s -a -u 0.005 -i %s -p %s -f %s >/dev/null" % (framerate, max_f0, min_f0, in_wav_file, out_est_file, out_fz_file)
    
    cmd =  _reaper_bin + " -s -e %s -x %s -m %s -a -u 0.005 -i %s -p %s -f %s >/dev/null" % (framerate, max_f0, min_f0, wavefile, out_est_file, out_fz_file)
    

    subprocess.call(cmd, shell=True)





if __name__=="__main__":

    main_work()

