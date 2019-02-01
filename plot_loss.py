#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
import fileinput
from argparse import ArgumentParser

from libutil import readlist
import numpy as np
import pylab as pl
def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-o', dest='outfile', required=True)
    a.add_argument('-l', dest='logfile', required=True)
    opts = a.parse_args()
    
    # ===============================================
    
    log = readlist(opts.logfile)
    log = [line.split('|') for line in log]
    log = [line[3].strip() for line in log if len(line) >=4]
    
    #validation = [line.replace('validation epoch ', '') for line in log if line.startswith('validation epoch')]
    #train = [line.replace('train epoch ', '') for line in log if line.startswith('validation epoch')]

    validation = [line.split(':')[1].strip().split(' ') for line in log if line.startswith('validation epoch')]
    train = [line.split(':')[1].strip().split(' ') for line in log if line.startswith('train epoch')]
    validation = np.array(validation, dtype=float)
    train = np.array(train, dtype=float)
    print train.shape
    print validation.shape

    pl.subplot(211)
    pl.plot(validation.flatten())
    pl.subplot(212)
    pl.plot(train[:,:4])
    pl.show()
if __name__=="__main__":

    main_work()

