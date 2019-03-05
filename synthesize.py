# -*- coding: utf-8 -*-
#! /usr/bin/env python2
'''
Based on code by kyubyong park at https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import os
import sys
import timeit 
from argparse import ArgumentParser

import numpy as np
#from scipy.io.wavfile import write
import soundfile

import tensorflow as tf

from tqdm import tqdm

from utils import plot_alignment
from utils import spectrogram2wav, durations_to_position
from utils import split_streams, magphase_synth_from_compressed 
from data_load import load_data
from architectures import Text2MelGraph, SSRNGraph, BabblerGraph
from libutil import safe_makedir, basename
from configuration import load_config


def start_clock(comment):
    print ('%s... '%(comment)),
    return (timeit.default_timer(), comment)

def stop_clock((start_time, comment), width=40):
    padding = (width - len(comment)) * ' '
    print ('%s--> took %.2f seconds' % (padding, (timeit.default_timer() - start_time)) )

def denorm(data, stats, type):
    if type=='minmax':
        mini = stats[0,:].reshape(1,-1)
        maxi = stats[1,:].reshape(1,-1)  
        X = data   
        # http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html
        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (maxi - mini) + mini   
        return X_scaled        
    elif type=='meanvar':
        mean = stats[0,:].reshape(1,-1)
        std = stats[1,:].reshape(1,-1)
        data = (data * std) + mean
    else:
        sys.exit('Unknown normtype: %s'%(type))
    return data

## TODO: compare efficiency  etc with encode_text + synth_codedtext2mel and possibly remove this version
def synth_text2mel(hp, L, g, sess, speaker_data=None, duration_data=None, \
                        labels=None, position_in_phone_data=None):
    '''
    L: texts
    g: synthesis graph
    sess: Session
    '''
    Y = np.zeros((len(L), hp.max_T, hp.n_mels), np.float32)
    prev_max_attentions = np.zeros((len(L),), np.int32)

    ### -- set up counters to detect & record sentence end, used for trimming and early stopping --
    ends = []  ## indices of first padding character after the last letter
    for i in range(len(L)):
        ends.append((np.where(L[i,:]==0)[0][0]))
    ends = np.array(ends)
    endcounts = np.zeros(ends.shape, dtype=int)  ## counts of the number of times attention has focussed (max) on these indices
    endcount_threshold = 1 ## number of times we require attention to focus on end before we consider synthesis finished
    t_ends = np.ones(ends.shape, dtype=int) * hp.max_T  ## The frame index when endcounts is sufficiently high, which we'll consider the end of the utterance
                                                        ## NB: initialised to max_T -- will default to this.

    t = start_clock('gen')
    feeddict = {g.L: L, g.mels: Y, g.prev_max_attentions: prev_max_attentions}
    if hp.multispeaker:
        feeddict[g.speakers] = speaker_data   
    if hp.use_external_durations:
        feeddict[g.durations] = duration_data  
    if hp.merlin_label_dir:
        feeddict[g.merlin_label] = labels  
    if 'position_in_phone' in hp.history_type:
        feeddict[g.position_in_phone] = position_in_phone_data


    for j in tqdm(range(hp.max_T)): # always run for max num of mel-frames
        _Y, _max_attentions, _alignments, = \
                    sess.run([ g.Y, g.max_attentions, g.alignments], feeddict)                                    

        #### OLDER VERSION (TODO - prune):
        # if hp.multispeaker:
        #     _Y, _max_attentions, _alignments, = \
        #         sess.run([ g.Y, g.max_attentions, g.alignments],
        #                  {g.L: L,
        #                   g.mels: Y,
        #                   g.speakers: speaker_data,
        #                   g.prev_max_attentions: prev_max_attentions}) ##
        # else:
        #     _Y, _max_attentions, _alignments, = \
        #         sess.run([ g.Y, g.max_attentions, g.alignments],
        #                  {g.L: L,
        #                   g.mels: Y,
        #                   g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop

        Y[:, j, :] = _Y[:, j, :]
        prev_max_attentions = _max_attentions[:, j]

        feeddict[g.mels] = Y
        feeddict[g.prev_max_attentions] = prev_max_attentions

        ## Work out if we've reach end of any/all sentences in batch:-
        reached_end = (_max_attentions[:, j] >= ends) ## is attention focussing on or beyond end of textual sentence?
        endcounts += reached_end
        for (i,(current, endcount)) in enumerate(zip(t_ends, endcounts)):
            if current == hp.max_T: ## if hasn't changed from initialisation value
                if endcount >= endcount_threshold:
                    t_ends[i] = j
        ## Bail out early if all sentences seem to be finished:
        if (t_ends < hp.max_T).all():
            print('finished here:')
            print(t_ends)
            break

    return (Y, t_ends.tolist())

def synth_babble(hp, g, sess, seed=False, nsamples=16):
    '''
    g: synthesis graph
    sess: Session
    TODO: always use random starting condition? Otherwise all samples are identical 
    '''
    assert not seed, 'TODO: implement seeding babbler'  

    Y = np.zeros((nsamples, hp.max_T, hp.n_mels), np.float32)
    
    t = start_clock('babbling')
    for j in tqdm(range(hp.max_T)):
        _Y, = sess.run([ g.Y], {g.mels: Y}) 
        Y[:, j, :] = _Y[:, j, :]
    return Y

def synth_codedtext2mel(hp, K, V, ends, g, sess, speaker_data=None, duration_data=None, \
                labels=None, position_in_phone_data=None):
    '''
    K, V: coded texts
    g: synthesis graph
    sess: Session
    '''
    Y = np.zeros((len(K), hp.max_T, hp.n_mels), np.float32) # note that len(K) == num_sentences that we want to generate wavs for
    alignments = np.zeros((len(ends), hp.max_N, hp.max_T), np.float32)
    prev_max_attentions = np.zeros((len(K),), np.int32)

    ### -- set up counters to detect & record sentence end, used for trimming and early stopping --

    endcounts = np.zeros(ends.shape, dtype=int)  ## counts of the number of times attention has focussed (max) on these indices
    endcount_threshold = 1 ## number of times we require attention to focus on end before we consider synthesis finished
    t_ends = np.ones(ends.shape, dtype=int) * hp.max_T  ## The frame index when endcounts is sufficiently high, which we'll consider the end of the utterance
                                                        ## NB: initialised to max_T -- will default to this.


    t = start_clock('gen')
    feeddict = {g.K: K, g.V: V, g.mels: Y, g.prev_max_attentions: prev_max_attentions}
    if hp.multispeaker:
        feeddict[g.speakers] = speaker_data  
    if hp.use_external_durations:
        feeddict[g.durations] = duration_data    
    if hp.merlin_label_dir:
        feeddict[g.merlin_label] = labels      
    if 'position_in_phone' in hp.history_type:
        feeddict[g.position_in_phone] = position_in_phone_data             
    for j in tqdm(range(hp.max_T)):  # always run for max num of mel-frames
        _Y, _max_attentions, _alignments, = \
                    sess.run([ g.Y, g.max_attentions, g.alignments], feeddict)                                    

        #### OLDER VERSION (TODO - prune):--
        # if hp.multispeaker:
        #     _Y, _max_attentions, _alignments, = \
        #         sess.run([ g.Y, g.max_attentions, g.alignments],
        #                  {g.K: K,
        #                   g.V: V,
        #                   g.mels: Y,
        #                   g.speakers: speaker_data,
        #                   g.prev_max_attentions: prev_max_attentions}) ##
        # else:
        #     _Y, _max_attentions, _alignments, = \
        #         sess.run([ g.Y, g.max_attentions, g.alignments],
        #                  {g.K: K,
        #                   g.V: V,
        #                   g.mels: Y,
        #                   g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop



        Y[:, j, :] = _Y[:, j, :] # build up mel-spec frame-by-frame
        alignments[:, :, j] = _alignments[:, :, j] # build up attention matrix frame-by-frame
        prev_max_attentions = _max_attentions[:, j]

        feeddict[g.mels] = Y
        feeddict[g.prev_max_attentions] = prev_max_attentions

        ## Work out if we've reach end of any/all sentences in batch:-
        reached_end = (_max_attentions[:, j] >= ends) ## is attention focussing on or beyond end of textual sentence?
        endcounts += reached_end
        for (i,(current, endcount)) in enumerate(zip(t_ends, endcounts)):
            if current == hp.max_T: ## if hasn't changed from initialisation value
                if endcount >= endcount_threshold:
                    t_ends[i] = j
        ## Bail out early if all sentences seem to be finished:
        if (t_ends < hp.max_T).all():
            print('finished here:')
            print(t_ends)
            break

    return (Y, t_ends.tolist(), alignments)

def encode_text(hp, L, g, sess, speaker_data=None, labels=None):  

    feeddict = {g.L: L}
    if hp.multispeaker:
        feeddict[g.speakers] = speaker_data   
    if hp.merlin_label_dir:
        feeddict[g.merlin_label] = labels  
    K, V = sess.run([ g.K, g.V], feeddict)
    return (K, V)

def get_text_lengths(L):
    ends = []  ## indices of first padding character after the last letter
    for i in range(len(L)):
        ends.append((np.where(L[i,:]==0)[0][0]))   ## TODO: have to go back to L to work this out?
    ends = np.array(ends)    
    return ends


def synth_mel2mag(hp, Y, g, sess, batchsize=128):
    #assert speaker_data==None  ## TODO: remove, or might speaker-condition SSRN at some point?
    
    if batchsize > 0:
        nbatches = max(1, len(Y) / batchsize)
        batches = np.array_split(Y, nbatches)
    else:
        batches = [Y]

    Z = np.concatenate([sess.run(g.Z, {g.mels: Y_batch}) for Y_batch in batches])
    return Z


def split_batch(synth_batch, end_indices):
    outputs = []
    for i, predmel in enumerate(synth_batch):
        length = end_indices[i]
        outputs.append(predmel[:length, :])
    return outputs

def make_mel_batch(hp, fnames, oracle=True): ## TODO: refactor with list2batch ?
    lengths = []
    if oracle:
        source = hp.coarse_audio_dir
        bases = [basename(fname) for fname in fnames]
        mels = [os.path.join(hp.coarse_audio_dir, base + '.npy') for base in bases]
    else:
        mels = fnames
    mels = [np.load(melfile) for melfile in mels] 
    mel_batch = np.zeros((len(mels), hp.max_T, hp.n_mels), np.float32)
    for (i,mel) in enumerate(mels):
        length,n = mel.shape
        mel_batch[i,:length,:] = mel
        lengths.append(length * hp.r)
    return mel_batch, lengths

def list2batch(inlist, pad_length):
    lengths = []
    m,dim = inlist[0].shape
    
    if pad_length==0:
        pad_length = max([a.shape[0] for a in inlist])

    batch = np.zeros((len(inlist), pad_length, dim), np.float32)
    for (i,array) in enumerate(inlist):
        length,n = array.shape
        assert length <= pad_length
        assert n==dim
        batch[i,:length,:] = array
    return batch


def restore_latest_model_parameters(sess, hp, model_type):
    model_types = {  't2m': 'Text2Mel', 
                    'ssrn': 'SSRN', 
                    'babbler': 'Text2Mel'
                  }  ## map model type to string used in scope
    scope = model_types[model_type]
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope)
    saver = tf.train.Saver(var_list=var_list)
    savepath = hp.logdir + "-" + model_type
    latest_checkpoint = tf.train.latest_checkpoint(savepath)
    if latest_checkpoint is None: sys.exit('No %s at %s?'%(model_type, savepath))
    latest_epoch = latest_checkpoint.strip('/ ').split('/')[-1].replace('model_epoch_', '')
    saver.restore(sess, latest_checkpoint)
    print("Model of type %s restored from latest epoch %s"%(model_type, latest_epoch))
    return latest_epoch


def babble(hp, num_sentences=0):

    if num_sentences == 0:
        num_sentences = 4 # default
    g1 = BabblerGraph(hp, mode="synthesize"); print("Babbler graph loaded")
    g2 = SSRNGraph(hp, mode="synthesize"); print("SSRN graph loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        babbler_epoch = restore_latest_model_parameters(sess, hp, 'babbler')
        ssrn_epoch = restore_latest_model_parameters(sess, hp, 'ssrn')

        t = start_clock('Babbling...')
        Y = synth_babble(hp, g1, sess, seed=False, nsamples=num_sentences)
        stop_clock(t)

        t = start_clock('Mel2Mag generating...')
        Z = synth_mel2mag(hp, Y, g2, sess)
        stop_clock(t) 

        if (np.isnan(Z).any()):  ### TODO: keep?
            Z = np.nan_to_num(Z)

        # Generate wav files
        outdir = os.path.join(hp.voicedir, 'synth_babble', '%s_%s'%(babbler_epoch, ssrn_epoch))
        safe_makedir(outdir)
        for i, mag in enumerate(Z):
            print("Applying Griffin-Lim to sample number %s"%(i))
            wav = spectrogram2wav(hp, mag)
            write(outdir + "/{:03d}.wav".format(i), hp.sr, wav)




def synthesize(hp, speaker_id='', num_sentences=0):
    assert hp.vocoder=='griffin_lim', 'Other vocoders than griffin_lim not yet supported'

    dataset = load_data(hp, mode="synthesis") #since mode != 'train' or 'validation', will load test_transcript rather than transcript
    fpaths, L = dataset['fpaths'], dataset['texts']
    position_in_phone_data = duration_data = labels = None # default
    if hp.use_external_durations:
        duration_data = dataset['durations']
        if num_sentences > 0:
            duration_data = duration_data[:num_sentences, :, :]

    if 'position_in_phone' in hp.history_type:
        ## TODO: combine + deduplicate with relevant code in train.py for making validation set
        def duration2position(duration, fractional=False):     
            ### very roundabout -- need to deflate A matrix back to integers:
            duration = duration.sum(axis=0)
            #print(duration)
            # sys.exit('evs')   
            positions = durations_to_position(duration, fractional=fractional)
            ###positions = end_pad_for_reduction_shape_sync(positions, hp)
            positions = positions[0::hp.r, :]         
            #print(positions)
            return positions

        position_in_phone_data = [duration2position(dur, fractional=('fractional' in hp.history_type)) \
                        for dur in duration_data]       
        position_in_phone_data = list2batch(position_in_phone_data, hp.max_T)



    # Ensure we aren't trying to generate more utterances than are actually in our test_transcript
    if num_sentences > 0:
        assert num_sentences < len(fpaths)
        L = L[:num_sentences, :]
        fpaths = fpaths[:num_sentences]

    bases = [basename(fpath) for fpath in fpaths]

    if hp.merlin_label_dir:
        labels = [np.load("{}/{}".format(hp.merlin_label_dir, basename(fpath)+".npy")) \
                              for fpath in fpaths ]
        labels = list2batch(labels, hp.max_N)


    if speaker_id:
        speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))
        speaker_ix = speaker2ix[speaker_id]

        ## Speaker codes are held in (batch, 1) matrix -- tiling is done inside the graph:
        speaker_data = np.ones((len(L), 1))  *  speaker_ix
    else:
        speaker_data = None

    # Load graph 
    ## TODO: generalise to combine other types of models into a synthesis pipeline?
    g1 = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
    g2 = SSRNGraph(hp, mode="synthesize"); print("Graph 2 (ssrn) loaded")

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        ### TODO: specify epoch from comm line?
        ### TODO: t2m and ssrn from separate configs?

        t2m_epoch = restore_latest_model_parameters(sess, hp, 't2m')
        ssrn_epoch = restore_latest_model_parameters(sess, hp, 'ssrn')

        # Pass input L through Text2Mel Graph
        t = start_clock('Text2Mel generating...')
        ### TODO: after futher efficiency testing, remove this fork
        if 1:  ### efficient route -- only make K&V once  ## 3.86, 3.70, 3.80 seconds (2 sentences)
            text_lengths = get_text_lengths(L)
            K, V = encode_text(hp, L, g1, sess, speaker_data=speaker_data, labels=labels)
            Y, lengths, alignments = synth_codedtext2mel(hp, K, V, text_lengths, g1, sess, \
                                speaker_data=speaker_data, duration_data=duration_data, \
                                position_in_phone_data=position_in_phone_data,\
                                labels=labels)
        else: ## 5.68, 5.43, 5.38 seconds (2 sentences)
            Y, lengths = synth_text2mel(hp, L, g1, sess, speaker_data=speaker_data, \
                                            duration_data=duration_data, \
                                            position_in_phone_data=position_in_phone_data, \
                                            labels=labels)
        stop_clock(t)

        ### TODO: useful to test this?
        # print(Y[0,:,:])
        # print (np.isnan(Y).any())
        # print('nan1')

        # Then pass output Y of Text2Mel Graph through SSRN graph to get high res spectrogram Z.
        t = start_clock('Mel2Mag generating...')
        Z = synth_mel2mag(hp, Y, g2, sess)
        stop_clock(t) 

        if (np.isnan(Z).any()):  ### TODO: keep?
            Z = np.nan_to_num(Z)

        # Generate wav files
        outdir = os.path.join(hp.sampledir, 't2m%s_ssrn%s'%(t2m_epoch, ssrn_epoch))
        if speaker_id:
            outdir += '_speaker-%s'%(speaker_id)
        safe_makedir(outdir)
        print("Generating wav files, will save to following dir: %s"%(outdir))
        for i, mag in enumerate(Z):
            print("Working on %s"%(bases[i]))
            mag = mag[:lengths[i]*hp.r,:]  ### trim to generated length
            
            if hp.vocoder=='magphase_compressed':
                mag = denorm(mag, s, hp.normtype)
                streams = split_streams(mag, ['mag', 'lf0', 'vuv', 'real', 'imag'], [60,1,1,45,45])
                wav = magphase_synth_from_compressed(streams, samplerate=hp.sr)
            elif hp.vocoder=='griffin_lim':                
                wav = spectrogram2wav(hp, mag)
            else:
                sys.exit('Unsupported vocoder type: %s'%(hp.vocoder))
            #write(outdir + "/{}.wav".format(bases[i]), hp.sr, wav)
            soundfile.write(outdir + "/{}.wav".format(bases[i]), wav, hp.sr)
            
        # Plot attention alignments 
        for i in range(num_sentences):
            plot_alignment(hp, alignments[i], utt_idx=i+1, t2m_epoch=t2m_epoch, dir=outdir)


def main_work():

    #################################################
      
    # ============= Process command line ============

    a = ArgumentParser()
    a.add_argument('-c', dest='config', required=True, type=str)
    a.add_argument('-speaker', default='', type=str)
    a.add_argument('-N', dest='num_sentences', default=0, type=int)
    a.add_argument('-babble', action='store_true')
    
    opts = a.parse_args()
    
    # ===============================================
    hp = load_config(opts.config)
    
    if hp.multispeaker:
        assert opts.speaker, 'Please specify a speaker from speaker_list with -speaker flag'
        assert opts.speaker in hp.speaker_list

    if opts.babble:
        babble(hp, num_sentences=opts.num_sentences)
    else:
        synthesize(hp, speaker_id=opts.speaker, num_sentences=opts.num_sentences)


if __name__=="__main__":

    main_work()
