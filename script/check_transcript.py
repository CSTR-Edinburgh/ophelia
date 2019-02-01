#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - May 2018 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk


import re
import os
import sys
import glob
import random
import numpy as np
import codecs

from argparse import ArgumentParser
from tqdm import tqdm

from libutil import readlist, writelist, basename

from bashplotlib.histogram import plot_hist




def main_work():


    #################################################
      
    # ======== Get stuff from command line ==========

    a = ArgumentParser()
    a.add_argument('-i', dest='infile', required=True) 
    a.add_argument('-cmp', dest='cmpdir', required=False, default='')          
    a.add_argument('-maxframes', dest='maxframes', type=int, default=10000000)
    a.add_argument('-maxletters', dest='maxletters', type=int, default=10000000)
    a.add_argument('-phone', action='store_true', help='Use the 4th field of transcript, containing space-delimited phones')
    a.add_argument('-speaker', action='store_true')
    a.add_argument('-o', dest='outfile', required=False, default='')  
  
    opts = a.parse_args()
    
    # ===============================================
    

    texts = codecs.open(opts.infile, 'r', 'utf-8', errors='ignore').readlines()
    texts = [line.strip('\n\r |') for line in texts]
    texts = [t for t in texts if t != '']
    texts = [line.strip().split("|") for line in texts]

    for line in texts:
        assert len(line) == len(texts[0]), line




    #print prompts

    all_frame_lengths = []
    all_letter_lengths = []

    frame_lengths = []
    letter_lengths = []
    outlines = []
    letter_inventory = {}
    speaker_ids = {}

    missing_audio = []
    for text in tqdm(texts):

        base, plaintext, normtext = text[:3]
        if opts.phone:
            assert len(text) >= 4
            symbols = text[3]
            symbols = re.split('\s+', symbols)
        else:
            symbols = list(normtext.lower())
        all_letter_lengths.append(len(symbols))        

        if opts.speaker:
            assert len(text) >= 5
            speaker_id = text[4]
            speaker_ids[speaker_id] = ''


        if opts.cmpdir:
            cmpfile = os.path.join(opts.cmpdir, base + '.npy')
            if not os.path.exists(cmpfile):
                missing_audio.append(base)
                continue
            nframe_audio, ndim = np.load(cmpfile).shape
            #print base
            
            all_frame_lengths.append(nframe_audio)
            if nframe_audio > opts.maxframes:
                continue
            frame_lengths.append(nframe_audio)


        if len(symbols) > opts.maxletters:
            continue

        
        letter_lengths.append(len(symbols))

        letter_inventory.update(dict(zip(symbols,symbols)))
        #print phones
        #outline = '%s||%s'%(base,  normtext)
        
        #outlines.append(outline)
        

    print 
    print '%s of %s sentences processed successfully -- to debug rerun ....'%(len(outlines), len(texts))
    print

    print 'missing audio for :'
    print missing_audio
    print
    print 
    print 'Keep %s of %s letter/phone tokens'%(len(letter_lengths), len(all_letter_lengths))
    print 'Keep %s of %s frames'%(len(frame_lengths), len(all_frame_lengths))
    print 
    
    if opts.phone:
        print '------------------'
        print 'Observed phones:'
        print '------------------'
        print 
        print sorted([str(k) for k in letter_inventory.keys()])
        print '------------------' 

        junctures = [p for p in letter_inventory.keys() if p.startswith(u'<') and p.endswith(u'>')]
        junctures.remove('<_START_>')
        junctures.remove('<_END_>')

        junctures = [p.strip('<>') for p in junctures]
        junctures = [j for j in junctures if j != '']
        print 
        print 
        print 'Junctures (many of these may be spurious):'
        print
        print ' '.join(sorted(junctures))
        print 

    else:
        print '------------------'
        print 'Observed letters:'
        print '------------------'
        print 
        print [''.join(sorted(letter_inventory.keys()))]
        print '------------------' 
    print 
    print 'Letter/phone length max:'
    print max(letter_lengths)

    if opts.speaker:
        print '------------------'
        print 'Observed speakers:'
        print '------------------'
        print 
        print sorted(speaker_ids.keys())
        print '------------------'         



    if opts.cmpdir:
        print 'Frame length max:'    
        print max(frame_lengths)

    

    ## 5% for testing:
    # outlines = np.array(outlines) ## for indexing with lists 
    # n_train = int(len(outlines) * 0.95)
    # ixx = range(len(outlines))
    # random.seed(12345)
    # random.shuffle(ixx) ## TODO seeds everywhere
    # train_ixx = sorted(ixx[:n_train])
    # test_ixx = sorted(ixx[n_train:])


    #writelist(outlines, opts.outfile)
    # writelist(outlines[test_ixx], opts.outfile + '.test')
    

    plot_hist(letter_lengths, xlab=True, bincount=30, title='Histogram of sentence length in letters', height=10, colour='k')

    if opts.cmpdir:
        plot_hist(frame_lengths, xlab=True, bincount=30, title='Histogram of sentence length in frames', height=10, colour='k')



    print
    print 'Run again with -maxframes and -maxletters to exclude the tail...'
    print 'note: configure hp.max_T to be maxframes/4'


    if opts.outfile:
        f = open(opts.outfile, 'w')
        for text in tqdm(texts):

            base = text[0]
            if base not in missing_audio:
                f.write('|'.join(text) + '\n')
        f.close()

if __name__=="__main__":

    main_work()

