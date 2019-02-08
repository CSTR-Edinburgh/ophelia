#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser
import codecs


def main_work():

    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='infile', required=True, \
                    help= "File in LJ speech transcription format: https://keithito.com/LJ-Speech-Dataset/")
    a.add_argument('-o', dest='outfile', required=True, \
                    help= "File in Festival utts.data scheme format")
    opts = a.parse_args()
    
    # ===============================================
    
    f = codecs.open(opts.infile, 'r', encoding='utf8')
    lines = f.readlines()
    f.close()

    f = codecs.open(opts.outfile, 'w', encoding='utf8')
    for line in lines:
        fields = line.strip('\n\r ').split('|')
        assert len(fields) >= 3
        name, _, text = fields[:3]
        text = text.replace('"', '\\"')
        f.write('(%s "%s")\n'%(name, text))
    f.close()




if __name__=="__main__":

    main_work()

