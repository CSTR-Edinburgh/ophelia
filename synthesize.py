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
from scipy.io.wavfile import write

import tensorflow as tf

from tqdm import tqdm

from utils import plot_alignment
from utils import spectrogram2wav
from utils import split_streams, magphase_synth_from_compressed 
from data_load import load_data, load_vocab
from architectures import Text2MelGraph, SSRNGraph, BabblerGraph
from libutil import safe_makedir, basename, load_config

import pickle 


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
def synth_text2mel(hp, L, g, sess, speaker_data=None):
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
    for j in tqdm(range(hp.max_T)): # always run for max num of mel-frames
        if hp.multispeaker:
            _Y, _max_attentions, _alignments, = \
                sess.run([ g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.speakers: speaker_data,
                          g.prev_max_attentions: prev_max_attentions}) ##
        else:
            _Y, _max_attentions, _alignments, = \
                sess.run([ g.Y, g.max_attentions, g.alignments],
                         {g.L: L,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop
        Y[:, j, :] = _Y[:, j, :]
        prev_max_attentions = _max_attentions[:, j]

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

def synth_codedtext2mel(hp, K, V, ends, g, sess, speaker_data=None):
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
    for j in tqdm(range(hp.max_T)):  # always run for max num of mel-frames
        if hp.multispeaker:
            _Y, _max_attentions, _alignments, = \
                sess.run([ g.Y, g.max_attentions, g.alignments],
                         {g.K: K,
                          g.V: V,
                          g.mels: Y,
                          g.speakers: speaker_data,
                          g.prev_max_attentions: prev_max_attentions}) ##
        else:
            _Y, _max_attentions, _alignments, = \
                sess.run([ g.Y, g.max_attentions, g.alignments],
                         {g.K: K,
                          g.V: V,
                          g.mels: Y,
                          g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop
        Y[:, j, :] = _Y[:, j, :] # build up mel-spec frame-by-frame
        alignments[:, :, j] = _alignments[:, :, j] # build up attention matrix frame-by-frame
        prev_max_attentions = _max_attentions[:, j]

        ##NOTE JASON - commented out below code as it for some reason fails when monotonic attention is turned off...
        ## Work out if we've reach end of any/all sentences in batch:-
        # reached_end = (_max_attentions[:, j] >= ends) ## is attention focussing on or beyond end of textual sentence?
        # endcounts += reached_end
        # for (i,(current, endcount)) in enumerate(zip(t_ends, endcounts)):
        #     if current == hp.max_T: ## if hasn't changed from initialisation value
        #         if endcount >= endcount_threshold:
        #             t_ends[i] = j
        # ## Bail out early if all sentences seem to be finished:
        # if (t_ends < hp.max_T).all():
        #     print('finished here:')
        #     print(t_ends)
        #     break

    return (Y, t_ends.tolist(), alignments)

def synth_codedtext2mel_gtruth(hp, K, V, gtruth_mels, ends, g, sess, speaker_data=None):
    '''
    K, V: coded texts
    g: synthesis graph
    sess: Session
    '''
    alignments = np.zeros((len(ends), hp.max_N, hp.max_T), np.float32)
    prev_max_attentions = np.zeros((len(K),), np.int32)

    ### -- set up counters to detect & record sentence end, used for trimming and early stopping --

    endcounts = np.zeros(ends.shape, dtype=int)  ## counts of the number of times attention has focussed (max) on these indices
    endcount_threshold = 1 ## number of times we require attention to focus on end before we consider synthesis finished
    t_ends = np.ones(ends.shape, dtype=int) * hp.max_T  ## The frame index when endcounts is sufficiently high, which we'll consider the end of the utterance
                                                        ## NB: initialised to max_T -- will default to this.

    t = start_clock('gen')
    for j in tqdm(range(hp.max_T)):  # always run for max num of mel-frames
        if hp.multispeaker:
            _max_attentions, _alignments, = \
                sess.run([ g.max_attentions, g.alignments],
                         {g.K: K,
                          g.V: V,
                          g.mels: gtruth_mels,
                          g.speakers: speaker_data,
                          g.prev_max_attentions: prev_max_attentions}) ##
        else:
            _max_attentions, _alignments, = \
                sess.run([ g.max_attentions, g.alignments],
                         {g.K: K,
                          g.V: V,
                          g.mels: gtruth_mels,
                          g.prev_max_attentions: prev_max_attentions}) ## osw: removed global_step from synth loop
        alignments[:, :, j] = _alignments[:, :, j] # build up attention matrix frame-by-frame
        prev_max_attentions = _max_attentions[:, j]

        ##NOTE JASON - commented out below code as it for some reason fails when monotonic attention is turned off...
        ## Work out if we've reach end of any/all sentences in batch:-
        # reached_end = (_max_attentions[:, j] >= ends) ## is attention focussing on or beyond end of textual sentence?
        # endcounts += reached_end
        # for (i,(current, endcount)) in enumerate(zip(t_ends, endcounts)):
        #     if current == hp.max_T: ## if hasn't changed from initialisation value
        #         if endcount >= endcount_threshold:
        #             t_ends[i] = j
        # ## Bail out early if all sentences seem to be finished:
        # if (t_ends < hp.max_T).all():
        #     print('finished here:')
        #     print(t_ends)
        #     break

    return alignments

def encode_text(hp, L, g, sess, speaker_data=None):  
    if hp.multispeaker:
        K, V = sess.run([ g.K, g.V], {g.L: L, g.speakers: speaker_data})
    else: 
        K, V = sess.run([ g.K, g.V], {g.L: L}) 
    return (K, V)

def get_text_lengths(L):
    ends = []  ## indices of first padding character after the last letter
    for i in range(len(L)):
        ends.append((np.where(L[i,:]==0)[0][0]))   ## TODO: have to go back to L to work this out?
    ends = np.array(ends)    
    return ends


def synth_mel2mag(hp, Y, g, sess, batchsize=0):
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

def make_mel_batch(hp, fnames, oracle=True):
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

    # Load data
    (fpaths, L) = load_data(hp, mode="synthesis") #since mode != 'train' or 'validation', will load test_transcript rather than transcript
    bases = [basename(fpath) for fpath in fpaths]

    # Load vocab used for getting phone sequence from our inputs (used for plotting on attention diagrams)
    _, idx2char = load_vocab(hp)

    # Also retrieve ground truth mels so that we can get attention plots for them
    mels = [np.load(hp.coarse_audio_dir + os.path.sep + basename(fpath)+'.npy') for fpath in fpaths]
    mels_array = np.zeros((len(mels), hp.max_T, hp.n_mels), np.float32) # create empty fixed size array to hold mels
    for i in range(len(mels)): # copy data into this fixed sized array
        mels_array[i, :mels[i].shape[0], :mels[i].shape[1]] = mels[i]
    mels = mels_array # rename for convenience

    # Ensure we aren't trying to generate more utterances than are actually in our test_transcript
    if num_sentences > 0:
        assert num_sentences < len(bases)
        L = L[:num_sentences, :]

    # Get the input sequence phones/letters for each utterance (used to annotate attention diagrams)
    L_chars = L.tolist()
    for row in range(L.shape[0]):
        for col in range(L.shape[1]):
            # print(L[row][col], idx2char[L[row][col]])
            L_chars[row][col] = idx2char[L[row][col]]
    # print(L_chars[0])

    if speaker_id:
        speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))
        speaker_ix = speaker2ix[speaker_id]

        ## Speaker codes are held in (batch, 1) matrix -- tiling is done inside the graph:
        speaker_data = np.ones((len(L), 1))  *  speaker_ix
    else:
        speaker_data = None

    # Load graph 
    ## TODO: generalise to combine other types of models into a synthesis pipeline?
    if hp.monotonic_attention:
        g1 = Text2MelGraph(hp, mode="synthesize"); print("Graph 1 (t2m) loaded")
    else:
        g1 = Text2MelGraph(hp, mode="synthesize_non_monotonic"); print("Graph 1 (t2m) loaded, without monotonic attention")
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
            K, V = encode_text(hp, L, g1, sess, speaker_data=None)
            Y, lengths, gen_alignments = synth_codedtext2mel(hp, K, V, text_lengths, g1, sess, speaker_data=speaker_data)
        else: ## 5.68, 5.43, 5.38 seconds (2 sentences)
            Y, lengths = synth_text2mel(hp, L, g1, sess, speaker_data=speaker_data)
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
        outdir = os.path.join(outdir, 'monotonic_attention') if hp.monotonic_attention else os.path.join(outdir, 'non-monotonic_attention')
        if speaker_id:
            outdir += '_speaker-%s'%(speaker_id)
        safe_makedir(outdir)
        wavdir = os.path.join(outdir, 'wav')
        safe_makedir(wavdir)
        print("Generating wav files, will save to following dir: %s"%(wavdir))
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
            write(wavdir + "/{}.wav".format(bases[i]), hp.sr, wav)

        # Plot synthesis attention alignments 
        plots_dir = os.path.join(outdir, 'gen_alignment_plots')
        safe_makedir(plots_dir)
        for i in range(len(gen_alignments)):
            # print(L_chars[i])
            basefilename = plot_alignment(hp, gen_alignments[i], chars=L_chars[i], utt_name=bases[i], t2m_epoch=t2m_epoch, monotonic=hp.monotonic_attention, ground_truth=False, dir=plots_dir)
        
        prev_max_attentions = np.zeros((len(K),), np.int32)

        # Plot ground truth mel attention alignments 
        plots_dir = os.path.join(outdir, 'gtruth_alignment_plots')
        safe_makedir(plots_dir)
        if hp.monotonic_attention:
            gtruth_alignments = synth_codedtext2mel_gtruth(hp, K, V, mels, text_lengths, g1, sess, speaker_data=speaker_data)
        else:
            return_values = sess.run([g1.alignments], 
                                     {g1.L: L, 
                                      g1.mels: mels}) 
            gtruth_alignments = return_values[0] # sess run returns a list, so unpack this list
        for i in range(len(gtruth_alignments)):
            basefilename = plot_alignment(hp, gtruth_alignments[i], chars=L_chars[i], utt_name=bases[i], t2m_epoch=t2m_epoch, monotonic=hp.monotonic_attention, ground_truth=True, dir=plots_dir)

        # Save attention matrices, phones, t_ends and other related information to disk
        print('Saving alignments and other data to disk using pickle')
        alignment_data_dir = os.path.join(outdir, 'alignment_data')
        safe_makedir(alignment_data_dir)
        for utt_name, gen_alignment, gtruth_alignment, L_char, length in zip(bases, gen_alignments, gtruth_alignments, L_chars, lengths):
            monotonic_str = 'monotonic' if hp.monotonic_attention else 'non-monotonic'
            pickle_file_name = '{}_{}_{}.pkl'.format(hp.config_name, monotonic_str, utt_name)
            pickle.dump({'gen_alignment':gen_alignment, 'gtruth_alignment':gtruth_alignment, 'input_chars':L_char, 'decoder_timesteps':length}, open(os.path.join(alignment_data_dir, pickle_file_name), 'wb'))

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
