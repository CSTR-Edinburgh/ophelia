# -*- coding: utf-8 -*-
#!/usr/bin/env python2
'''
Adapted from original code by kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

from __future__ import print_function

import numpy as np
import tensorflow as tf
from utils import *
import codecs
import re
import os
import unicodedata

from libutil import basename, read_floats_from_8bit

import logging 

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

def load_data(hp, mode="train", get_speaker_codes=False, n_utts=0):
    '''Loads data
      Args:
          mode: "train" / "validation" / "synthesize".
    '''
    logging.info('Start loading data in mode: %s'%(mode))

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

    too_long_count_frames = 0
    too_long_count_text = 0
    no_data_count = 0

    for line in tqdm(lines, desc='load_data'):
        line = line.strip('\n\r |')
        if line == '':
            continue
        fields = line.strip().split("|")
        assert len(fields) >= 3,  fields
        fname, unnorm_text, norm_text = fields[:3]

        if mode in ["train", "validation"] and os.path.exists(hp.coarse_audio_dir):
            mel = "{}/{}".format(hp.coarse_audio_dir, fname+".npy")
            if not os.path.exists(mel):
                logging.debug('no file %s'%(mel))
                no_data_count += 1
                continue
            nframes = np.load(mel).shape[0]
            if nframes > hp.max_T:
                #print('number of frames for %s is %s, exceeds max_T %s: skip it'%(fname, nframes, hp.max_T))
                too_long_count_frames += 1
                continue

        if len(fields) >= 4:
            phones = fields[3]

        if hp.validpatt: 
            if mode=="train": 
                if hp.validpatt in fname:
                    continue
            elif mode=="validation":
                if hp.validpatt not in fname:
                    continue

        if hp.input_type == 'phones':
            phones = phones_normalize(phones, char2idx) # in case of phones, all EOS markers are assumed included
            letters_or_phones = [char2idx[char] for char in phones]
        elif hp.input_type == 'letters':
            text = text_normalize(norm_text, hp) + "E"  # E: EOS
            letters_or_phones = [char2idx[char] for char in text]

        text_length = len(letters_or_phones)

        if text_length > hp.max_N:
            #print('number of letters/phones for %s is %s, exceeds max_N %s: skip it'%(fname, text_length, hp.max_N))
            too_long_count_text += 1
            continue

        texts.append(np.array(letters_or_phones, np.int32))

        fpath = os.path.join(hp.waveforms, fname + ".wav")
        fpaths.append(fpath)
        text_lengths.append(text_length)

        if get_speaker_codes:
            assert len(fields) >= 5
        if len(fields) >= 5:
            speaker = fields[4]
            speaker_ix = speaker2ix[speaker]
            speakers.append(np.array(speaker_ix, np.int32))                    

    logging.info ('Loaded data for %s sentences'%(len(texts)))
    logging.info ('Sentences skipped with missing features: %s'%(no_data_count))    
    logging.info ('Sentences skipped with > max_T (%s) frames: %s'%(hp.max_T, too_long_count_frames))
    logging.info ('Additional sentences skipped with > max_N (%s) letters/phones: %s'%(hp.max_N, too_long_count_text))
 

    if mode == 'train':
        texts = [text.tostring() for text in texts]  
        if get_speaker_codes:
            speakers = [speaker.tostring() for speaker in speakers]         
        if n_utts > 0:
            assert hp.n_utts <= len(fpaths)
            logging.info ('Take first %s (n_utts) sentences'%(n_utts))
            if get_speaker_codes:
                return fpaths[:n_utts], text_lengths[:n_utts], texts[:n_utts], speakers[:n_utts]
            else:
                return fpaths[:n_utts], text_lengths[:n_utts], texts[:n_utts]
        else:  
            if get_speaker_codes:
                return fpaths, text_lengths, texts, speakers
            else:  
                return fpaths, text_lengths, texts
    elif mode=='validation':
        #texts = [text for text in texts if len(text) <= hp.max_N]
        stacked_texts = np.zeros((len(texts), hp.max_N), np.int32)
        for i, text in enumerate(texts):
            stacked_texts[i, :len(text)] = text
        if get_speaker_codes:
            return fpaths, stacked_texts, speakers
        else:
            return fpaths, stacked_texts
    else:
        assert mode=='synthesis'
        stacked_texts = np.zeros((len(texts), hp.max_N), np.int32)
        for i, text in enumerate(texts):
            stacked_texts[i, :len(text)] = text
        return (fpaths, stacked_texts) ## fpaths only a way to get bases -- wav files probably do not exist



def get_batch(hp, batchsize, get_speaker_codes=False, n_utts=0):
    """Loads training data and put them in queues"""
    # print ('get_batch')
    with tf.device('/cpu:0'):
        # Load data
        if get_speaker_codes:
            fpaths, text_lengths, texts, speakers = load_data(hp, get_speaker_codes=True, n_utts=n_utts) 
        else:
            fpaths, text_lengths, texts = load_data(hp, n_utts=n_utts) 

        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // batchsize

        # Create Queues & parse
        if get_speaker_codes:
            fpath, text_length, text, speaker = tf.train.slice_input_producer([fpaths, text_lengths, texts, speakers], shuffle=True)
            speaker = tf.decode_raw(speaker, tf.int32)
        else:   
            fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)
        text = tf.decode_raw(text, tf.int32)  # (None,)

        if hp.random_reduction_on_the_fly:

            assert os.path.isdir(hp.full_mel_dir)
            def _load_and_reduce_spectrograms(fpath):
                fname = os.path.basename(fpath)
                melfile = "{}/{}".format(hp.full_mel_dir, fname.replace("wav", "npy"))
                magfile = "{}/{}".format(hp.full_audio_dir, fname.replace("wav", "npy"))

                mel = np.load(melfile)
                mag = np.load(magfile)

                start = np.random.randint(0, hp.r)

                mel =  mel[start::4, :]
                ### How it works:
                # >>> mel = np.arange(40)
                # >>> print mel[::4]
                # [ 0  4  8 12 16 20 24 28 32 36]
                # >>> print mel[0::4]
                # [ 0  4  8 12 16 20 24 28 32 36]
                # >>> print mel[1::4]
                # [ 1  5  9 13 17 21 25 29 33 37]
                # >>> print mel[2::4]
                # [ 2  6 10 14 18 22 26 30 34 38]
                # >>> print mel[3::4]
                # [ 3  7 11 15 19 23 27 31 35 39]

                ### need to pad end of mag accordingly (and trim start) so that it matches:--
                mag = np.pad(mag, [[0, start], [0, 0]], mode="constant")[start:,:]
                return fname, mel, mag

            fname, mel, mag = tf.py_func(_load_and_reduce_spectrograms, [fpath], [tf.string, tf.float32, tf.float32])

            mel_reduced = mel[::hp.r, :]

        elif hp.prepro:
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

        if hp.attention_guide_dir:
            def load_attention(fpath):
                attention_guide_file = "{}/{}".format(hp.attention_guide_dir, basename(fpath)+".npy")
                attention_guide = read_floats_from_8bit(attention_guide_file)
                return fpath, attention_guide
            _, attention_guide = tf.py_func(load_attention, [fpath], [tf.string, tf.float32]) # py_func wraps a python function and use it as a TensorFlow op.


        # Add shape information
        fname.set_shape(())
        text.set_shape((None,))
        if get_speaker_codes:
            speaker.set_shape((None,))
        if hp.attention_guide_dir:
            attention_guide.set_shape((None,None))  ## will be letters x frames
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.full_dim))

        # Batching
        tensordict = {'text': text, 'mel': mel, 'mag': mag, 'fname': fname}
        
        if get_speaker_codes:
            tensordict['speaker'] = speaker  
        if hp.attention_guide_dir:
            tensordict['attention_guide'] = attention_guide

        _, batched_tensor_dict = tf.contrib.training.bucket_by_sequence_length(             
                                            input_length=text_length,
                                            tensors=tensordict,
                                            batch_size=batchsize,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=batchsize*4,
                                            dynamic_pad=True)

        batched_tensor_dict['num_batch'] = num_batch        
        return batched_tensor_dict




