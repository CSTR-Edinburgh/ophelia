#!/usr/bin/env python
# -*- coding: utf-8 -*-
## Project: SCRIPT - February 2019 
## Contact: Oliver Watts - owatts@staffmail.ed.ac.uk
  
import sys
import os
import glob
import os
from argparse import ArgumentParser
from tqdm import tqdm
import  numpy as np
from scipy.spatial.distance import cdist 
from pathlib import Path
import codecs 
import re

def merlin_state_label_to_phone(labfile, outlabfile):
    labels = np.loadtxt(labfile, dtype=str, comments=None) ## default comments='#' breaks
    starts = labels[:,0].astype(int)[::5].reshape(-1,1)
    ends = labels[:,1].astype(int)[4::5].reshape(-1,1)
    fc = labels[:,2][::5]
    fc = np.array([line.replace('[2]','') for line in fc]).reshape(-1,1)
    phone_label = np.hstack([starts, ends, fc])
    np.savetxt(outlabfile, phone_label, fmt='%s')

def merlin_state_label_to_monophones(labfile):
    labels = np.loadtxt(labfile, dtype=str, comments=None) ## default comments='#' breaks
    starts = labels[:,0].astype(int)[::5]
    ends = labels[:,1].astype(int)[4::5]
    fc = labels[:,2][::5]
    mono = [label.split('-')[1].split('+')[0] for label in fc]
    # return zip(starts, ends, mono)
    lengths = (ends - starts) / 10000 ## length in msec
    return (mono, lengths)

def plain_phone_label_to_monophones(labfile):
    labels = np.loadtxt(labfile, dtype=str, comments=None) ## default comments='#' breaks
    starts = labels[:,0].astype(int)
    ends = labels[:,1].astype(int)
    mono = labels[:,2]
    lengths = (ends - starts) / 10000 ## length in msec
    return (mono, lengths)

def match_up(merlin_label_timings, dctts_label):
    merlin_silence_symbols = ['pau', 'sil', 'skip']
    merlin_label, merlin_timings = merlin_label_timings
    output = []
    timings = []
    m = d = 0
    # print '====='
    # print merlin_label
    # print dctts_label
    while m < len(merlin_label) and d < len(dctts_label):
        # print (m,d)
        if merlin_label[m] in merlin_silence_symbols:
            assert dctts_label[d].startswith('<'), (dctts_label[d], merlin_label[m])
            timings.append(merlin_timings[m])
            m += 1
            d += 1
        else:
            if dctts_label[d].startswith('<'):
                timings.append(0)
                d += 1
            else:
                timings.append(merlin_timings[m])
                m += 1
                d += 1
    assert m==len(merlin_label)
    while d < len(dctts_label): ## in case punctuation then <_END_> at end of dctts
        timings.append(0)
        d += 1
    return timings


def resample_timings(lengths, from_rate=5.0, to_rate=12.5, total_duration=0):
    '''
    lengths: array of durations in msec. Each value is divisible by from_rate.
    Return converted sequence where values are divisible by to_rate.
    If total_duration, increase length of *last* item to match this total_duration.
    '''
    assert (lengths % from_rate).all() == 0.0

    ## find closest valid end given new sample rate
    ends = np.cumsum(lengths)
    new_valid_positions = np.arange(0,ends[-1] + (from_rate*3),float(to_rate))
    distances = cdist(np.expand_dims(ends,-1), np.expand_dims(new_valid_positions, -1))
    in_new_rate = new_valid_positions[distances.argmin(axis=1)]
    if 0:
        print zip(ends, in_new_rate)

    ## undo cumsum to get back from end points to durations:
    in_new_rate[1:] -= in_new_rate[:-1].copy()

    if total_duration:
        diff = total_duration - in_new_rate.sum()
        assert in_new_rate.sum() <= total_duration, (in_new_rate.sum(), total_duration)
        assert diff % to_rate == 0.0
        in_new_rate[-1] += diff

    return in_new_rate



def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-t', dest='transcript_file', required=True)    
    a.add_argument('-o', dest='outfile', required=True)
    a.add_argument('-l', dest='labdir', required=True)   
    a.add_argument('-f', dest='featdir', required=False, default='')   

    a.add_argument('-ir', dest='inrate', type=float, default=5.0) 
    a.add_argument('-or', dest='outrate', type=float, default=12.5)   

    a.add_argument('-plain', dest='plain_phone_label', action='store_true')

    opts = a.parse_args()
    
    # ===============================================

    transcript = read_transcript(opts.transcript_file)
    labdir = Path(opts.labdir)
    if opts.featdir:
        training = True ## train time, have acoustics whose length the label must match
        featdir = Path(opts.featdir)
        featfiles = set([fname.stem for fname in featdir.glob('*.npy')])
    else:
        training = False ## durations in labels are synthetic
    outfile = opts.outfile

    if sum([1 for i in labdir.glob('*.lab')]) == 0:
        sys.exit('No label files in %s'%(opts.labdir))
    
    for labfile in tqdm(sorted(labdir.glob('*.lab'))):

        if labfile.stem not in transcript:          
            continue
        if training:
            if labfile.stem not in featfiles:
                continue
        if opts.plain_phone_label:
            (mono, lengths) = plain_phone_label_to_monophones(labfile)
        else:
            (mono, lengths) = merlin_state_label_to_monophones(labfile)
        label_msec_length = lengths.sum() 

        if training:
            features = np.load((featdir / labfile.stem).with_suffix('.npy'))
            audio_msec_length = features.shape[0] * opts.outrate
            resampled_lengths = resample_timings(lengths, from_rate=opts.inrate, to_rate=opts.outrate, total_duration=audio_msec_length)
        else:
            resampled_lengths = resample_timings(lengths, from_rate=opts.inrate, to_rate=opts.outrate)
        resampled_lengths_in_frames = (resampled_lengths / opts.outrate).astype(int)
        
        timings = match_up((mono, resampled_lengths_in_frames), transcript[labfile.stem]['phones'])
        assert len(transcript[labfile.stem]['phones']) == len(timings), (len(transcript[labfile.stem]['phones']), len(timings), transcript[labfile.stem]['phones'], timings)
        transcript[labfile.stem]['duration'] = timings

    write_transcript(transcript, outfile, duration=True)

def read_transcript(transcript_file):
    texts = codecs.open(transcript_file, 'r', 'utf-8', errors='ignore').readlines()
    texts = [line.strip('\n\r |') for line in texts]
    texts = [t for t in texts if t != '']
    texts = [line.strip().split("|") for line in texts]

    for line in texts:
        assert len(line) == len(texts[0]), line

    transcript = {}

    for text in tqdm(texts):
        assert len(text) >= 4  ## assume phones
        base, plaintext, normtext, phones = text[:4]
        #symbols = text[3]
        phones = re.split('\s+', phones)
        transcript[base] = {'phones': phones, 'text': normtext}

    return transcript

def write_transcript(texts, transcript_file, duration=False):

    f = codecs.open(transcript_file, 'w', 'utf-8')

    for base in tqdm(sorted(texts.keys())):
        phones = ' '.join(texts[base]['phones'])
        line = '%s||%s|%s'%(base, texts[base]['text'], phones)
        if duration:
            if 'duration' not in texts[base]:
                print 'Warning: skip %s because no duration'%(base)
                continue
            dur = ' '.join(np.array(texts[base]['duration'], dtype='str'))
            line += '||%s'%(dur)  ## leave empty speaker ID field 
        f.write(line + '\n')
    f.close()
    print 'Wrote to ' + transcript_file





if __name__=="__main__":
    main_work()

