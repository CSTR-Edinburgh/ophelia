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
    a.add_argument('-i', dest='indir', required=True)  ## mels   
    a.add_argument('-of', dest='outdir_f', required=True, \
                    help= "Put output f0 here: make it if it doesn't exist") 
    a.add_argument('-om', dest='outdir_m', required=True, \
                    help= "Put output mels here: make it if it doesn't exist")     
    a.add_argument('-f', dest='fzdir', required=True)  
    # a.add_argument('-framerate', required=False, default=0.005, type=float, help='rate in seconds for F0 track frames')
    # a.add_argument('-pattern', default='', \
    #                 help= "If given, only normalise files whose base contains this substring")
    a.add_argument('-ncores', default=1, type=int)
    #a.add_argument('-waveformat', default=False, action='store_true', help='call sox to format data (16 bit ).')    
    
    # a.add_argument('-twopass', default=False, action='store_true', help='Run initially on a subset of data to guess sensible limits, then run again. Assumes all data is from same speaker.')    
    opts = a.parse_args()
    
    # ===============================================
    
    for direc in [opts.outdir_f, opts.outdir_m]:
        if not os.path.isdir(direc):
            os.makedirs(direc)

    flist = sorted(glob.glob(opts.indir + '/*.npy'))

    print flist
    # print 'Extract with range %s %s'%(min_f0, max_f0)
    executor = ProcessPoolExecutor(max_workers=opts.ncores)
    futures = []
    for mel_file in flist:
        futures.append(executor.submit(
            partial(process, mel_file, opts.fzdir, opts.outdir_f, opts.outdir_m)))
    return [future.result() for future in tqdm(futures)]


def put_speech(m_data, filename):
    m_data = np.array(m_data, 'float32') # Ensuring float32 output
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return 



def process(mel_file, fzdir, outdir_f, outdir_m):
    _, base = os.path.split(mel_file)
    base = base.replace('.npy', '')

    mels = np.load(mel_file)
    fz_file = os.path.join(fzdir, base + '.f0')

    fz = read_est_file(fz_file)[:,2] # .reshape(-1,1)

    m,_ = mels.shape
    f = fz.shape[0]


    fz[fz<0.0] = 0.0

    if m > f:
        diff = m - f
        fz = np.pad(fz, (0,diff), 'constant').reshape(-1,1)

    put_speech(fz, os.path.join(outdir_f, base+'.f0'))
    put_speech(mels, os.path.join(outdir_m, base+'.mfbsp'))

    # print fz.shape
    # print mels.shape
    # print fz





    # out_file = os.path.join(outdir, base + '.pm')

    # in_wav_file = os.path.join(fzdir, base + '_tmp.wav')  ### !!!!!
    # cmd = 'sox %s -r 16000 %s '%(wavefile, in_wav_file)
    # subprocess.call(cmd, shell=True)

    # out_fz_file = os.path.join(fzdir, base + '.f0')
    # cmd =  _reaper_bin + " -s -e %s -x %s -m %s -a -u 0.005 -i %s -p %s -f %s >/dev/null" % (framerate, max_f0, min_f0, in_wav_file, out_est_file, out_fz_file)
    # subprocess.call(cmd, shell=True)





if __name__=="__main__":

    main_work()

