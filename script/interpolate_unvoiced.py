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

import scipy.interpolate

import  numpy as np
#from generate import read_est_file

empty_array = np.zeros((0,0)) # TODO: const


def interpolate_through_unvoiced(data, vuv=empty_array):

    assert len(data.shape) == 2, 'interpolate_through_unvoiced only accepts 2D arrays'
    if vuv.size == empty_array.size:
        assert data.shape[1] == 1, 'To find voicing from the data itself, use data with only a single channel'
        voiced_ix = np.where( data > 0.0 )[0]  ## equiv to np.nonzero(y)
    else:
        voiced_ix = np.where( vuv > 0.0 )[0]
    mean_voiced = data[voiced_ix, ...].mean(axis=0)  ## using fill_value='extrapolate' creates very extreme values where there are long initial/final silences
           ### TODO: this seems to affect denormalisation rather than training, look at extracintg stats and even training without regard to interpolated values?
    interpolator = scipy.interpolate.interp1d(voiced_ix, data[voiced_ix, ...], kind='linear', \
                                                axis=0, bounds_error=False, fill_value=mean_voiced)
    data_interpolated = interpolator(np.arange(data.shape[0])) # .reshape((-1,1)))

    voicing_flag = np.zeros((data.shape[0], 1))
    voicing_flag[voiced_ix] = 1.0

    return (data_interpolated, voicing_flag)




# def main_work():

#     #################################################
      
#     # ======== Get stuff from command line ==========

#     a = ArgumentParser()
#     a.add_argument('-i', dest='indir', required=True)    
#     a.add_argument('-o', dest='outdir', required=True, \
#                     help= "Put output here: make it if it doesn't exist")
#     a.add_argument('-f', dest='fzdir', required=True, \
#                     help= "Put F0 traces here: make it if it doesn't exist")    
#     a.add_argument('-pattern', default='', \
#                     help= "If given, only normalise files whose base contains this substring")
#     a.add_argument('-ncores', default=1, type=int)
#     a.add_argument('-twopass', default=False, action='store_true', help='Run initially on a subset of data to guess sensible limits, then run again. Assumes all data is from same speaker.')    
#     opts = a.parse_args()
    
#     # ===============================================
    
#     for direc in [opts.outdir, opts.fzdir]:
#         if not os.path.isdir(direc):
#             os.makedirs(direc)

#     flist = sorted(glob.glob(opts.indir + '/*.wav'))
    
#     if opts.twopass:
#         temp_outdir = opts.outdir.rstrip('/ ') + '_temp'
#         temp_fzdir = opts.fzdir.rstrip('/ ') + '_temp'
#         for direc in [temp_outdir, temp_fzdir]:
#             if not os.path.isdir(direc):
#                 os.makedirs(direc)

#         ## first pass with no min and max:
#         executor = ProcessPoolExecutor(max_workers=opts.ncores)
#         futures = []

#         ## first pass on small subset of data only
#         if flist > 100:
#             short_flist = flist[:100]
#         else:
#             short_flist = flist
#         for wave_file in short_flist:
#             futures.append(executor.submit(
#                 partial(process, wave_file, temp_outdir, temp_fzdir, pattern=opts.pattern)))
#         proc_list =  [future.result() for future in tqdm(futures)]

#         ## compute thresholds:
#         fzero_flist = glob.glob(temp_fzdir + '/*.f0')
#         all_voiced = []
#         for fname in tqdm(fzero_flist):
#             d = read_est_file(fname)
#             fz = d[:,2]
#             voiced = fz[fz > 0.0]
#             all_voiced.append(voiced)

#         all_voiced = np.hstack(all_voiced)
            
#         # De Looze and Rauzy (2009):
#         min_f0 = int((np.percentile(all_voiced, 35) * 0.72) - 10.0)
#         max_f0 = int((np.percentile(all_voiced, 65) * 1.9) + 10.0)

#     else:
#         min_f0 = 50        
#         max_f0 = 400

#     print 'Extract with range %s %s'%(min_f0, max_f0)
#     executor = ProcessPoolExecutor(max_workers=opts.ncores)
#     futures = []
#     for wave_file in flist:
#         futures.append(executor.submit(
#             partial(process, wave_file, opts.outdir, opts.fzdir, pattern=opts.pattern, min_f0=min_f0, max_f0=max_f0)))
#     return [future.result() for future in tqdm(futures)]






# def process(wavefile, outdir, fzdir, pattern='', max_f0=400, min_f0=50):
#     _, base = os.path.split(wavefile)
#     base = base.replace('.wav', '')
    
#     if pattern:
#         if pattern not in base:
#             return

#     out_est_file = os.path.join(outdir, base + '.pm')
#     out_fz_file = os.path.join(fzdir, base + '.f0')
#     cmd =  _reaper_bin + " -s -x %s -m %s -a -u 0.005 -i %s -p %s -f %s >/dev/null" % (max_f0, min_f0, wavefile, out_est_file, out_fz_file)
#     subprocess.call(cmd, shell=True)





if __name__=="__main__":
    sys.exit('TODO: CLI!!!')
    #main_work()

