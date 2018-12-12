# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

# from hyperparams import Hyperparams as hp
import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata

from tqdm import tqdm

def load_vocab(hp):
    char2idx = {char: idx for idx, char in enumerate(hp.vocab)}
    idx2char = {idx: char for idx, char in enumerate(hp.vocab)}
    return char2idx, idx2char

def text_normalize(text, hp):
    text = ''.join(char for char in unicodedata.normalize('NFD', text)
                           if unicodedata.category(char) != 'Mn') # Strip accents

    text = text.lower()
    text = re.sub("[^{}]".format(hp.vocab), " ", text)
    text = re.sub("[ ]+", " ", text)
    return text

def phones_normalize(text, char2idx):
    phones = re.split('\s+', text.strip(' \n'))
    for phone in phones: 
        if phone not in char2idx:
            print(text)
            sys.exit('Phone %s not listed in phone set'%(phone))
    return phones

def load_data(hp, mode="train", get_speaker_codes=False):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab(hp)

    if mode in ["train", "validation"]:
        transcript = os.path.join(hp.transcript)
    else:
        transcript = os.path.join(hp.test_transcript)

    if hp.multispeaker:
        speaker2ix = dict(zip(hp.speaker_list, range(len(hp.speaker_list))))

    fpaths, text_lengths, texts, speakers = [], [], [], []
    lines = codecs.open(transcript, 'r', 'utf-8').readlines()
    for line in tqdm(lines, desc='load_data'):
        line = line.strip('\n\r |')
        if line == '':
            continue
        fields = line.strip().split("|")
        assert len(fields) >= 3,  fields
        fname, unnorm_text, norm_text = fields[:3]
        if len(fields) >= 4:
            phones = fields[3]

        if hp.validpatt: 
            if mode=="train": 
                if hp.validpatt in fname:
                    continue
            elif mode=="validation":
                if hp.validpatt not in fname:
                    continue

        fpath = os.path.join(hp.waveforms, fname + ".wav")
        fpaths.append(fpath)
        #fnames.append(fname)

        if hp.input_type == 'phones':
            phones = phones_normalize(phones, char2idx) # in case of phones, all EOS markers are assumed included
            #ophones = phones
            phones = [char2idx[char] for char in phones]
            text_length = len(phones)
            texts.append(np.array(phones, np.int32))
        elif hp.input_type == 'letters':
            text = text_normalize(norm_text, hp) + "E"  # E: EOS
            text = [char2idx[char] for char in text]
            text_length = len(text)
            texts.append(np.array(text, np.int32))                    
        text_lengths.append(text_length)

        if get_speaker_codes:
            assert len(fields) >= 5
        if len(fields) >= 5:
            speaker = fields[4]
            speaker_ix = speaker2ix[speaker]
            #### speaker_ix = [speaker_ix] * text_length
            speakers.append(np.array(speaker_ix, np.int32))                    


    # if mode=="train":  
    #     if hp.validpatt: 
    #         #ys.exit('srverbetbnrt')
    #         texts = [text for (fname, text) in zip(fnames,texts) if hp.validpatt not in fname]   
    # elif mode=="validation":
    #     assert hp.validpatt, "Must specify hp.validpatt to use mode=='dev'"
    #     texts = [text for (fname, text) in zip(fnames, texts) if hp.validpatt in fname] 

    if mode == 'train':
        texts = [text.tostring() for text in texts]  
        if get_speaker_codes:
            speakers = [speaker.tostring() for speaker in speakers]         
        if hp.n_utts > 0:
            assert hp.n_utts <= len(fpaths)
            if get_speaker_codes:
                return fpaths[:hp.n_utts], text_lengths[:hp.n_utts], texts[:hp.n_utts], speakers[:hp.n_utts]
            else:
                return fpaths[:hp.n_utts], text_lengths[:hp.n_utts], texts[:hp.n_utts]
        else:  
            if get_speaker_codes:
                return fpaths, text_lengths, texts, speakers
            else:  
                return fpaths, text_lengths, texts
    elif mode=='validation':
        texts = [text for text in texts if len(text) <= hp.max_N]
        stacked_texts = np.zeros((len(texts), hp.max_N), np.int32)
        for i, text in enumerate(texts):
            stacked_texts[i, :len(text)] = text
        if get_speaker_codes:
            return fpaths, stacked_texts, speakers
        else:
            return fpaths, stacked_texts
    else:
        assert mode=='synthesis'
        #texts = texts[:5]  ### TODO: nsynth
        stacked_texts = np.zeros((len(texts), hp.max_N), np.int32)
        for i, text in enumerate(texts):
            stacked_texts[i, :len(text)] = text
        return (fpaths, stacked_texts) ## fpaths only a way to get bases -- wav files probably do not exist



def get_batch(hp, num=1, get_speaker_codes=False):
    """Loads training data and put them in queues"""
    # print ('get_batch')
    with tf.device('/cpu:0'):
        # Load data
        if get_speaker_codes:
            fpaths, text_lengths, texts, speakers = load_data(hp, get_speaker_codes=True) 
        else:
            fpaths, text_lengths, texts = load_data(hp) 

        maxlen, minlen = max(text_lengths), min(text_lengths)

        if num==1:
            batchsize = hp.B1                
        else:
            batchsize = hp.B2

        # Calc total batch count
        num_batch = len(fpaths) // batchsize

        # Create Queues & parse
        if get_speaker_codes:
            fpath, text_length, text, speaker = tf.train.slice_input_producer([fpaths, text_lengths, texts, speakers], shuffle=True)
            speaker = tf.decode_raw(speaker, tf.int32)
        else:   
            fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.prepro:
            def _load_spectrograms(fpath):
                fname = os.path.basename(fpath)
                mel = "{}/{}".format(hp.coarse_audio_dir, fname.replace("wav", "npy"))
                mag = "{}/{}".format(hp.full_audio_dir, fname.replace("wav", "npy"))
                if 0:
                    print ('mag file:')
                    print (mag)
                    print (np.load(mag).shape)
                return fname, np.load(mel), np.load(mag)

            fname, mel, mag = tf.py_func(_load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])
        else:
            fname, mel, mag = tf.py_func(load_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])  # (None, n_mels)

        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        if get_speaker_codes:
            speaker.set_shape((None,))
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.full_dim))
        #mag.set_shape((None, hp.n_fft//2+1))  ### OSW: softcoded this

        # Batching
        if get_speaker_codes:
            _, (texts, speakers, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, speaker, mel, mag, fname],
                                            batch_size=batchsize,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=batchsize*4,
                                            dynamic_pad=True)
        else:
            _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=batchsize,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=batchsize*4,
                                            dynamic_pad=True)
    if get_speaker_codes:
        return texts, speakers, mels, mags, fnames, num_batch
    else:
        return texts, mels, mags, fnames, num_batch

