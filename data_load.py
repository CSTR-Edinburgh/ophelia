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
            sys.exit('Phone %s not listed in phone set'%(phone))
    return phones

def load_data(hp, mode="train"):
    '''Loads data
      Args:
          mode: "train" or "synthesize".
    '''
    # Load vocabulary
    char2idx, idx2char = load_vocab(hp)

    if mode=="train":

        fpaths, text_lengths, texts = [], [], []
        transcript = os.path.join(hp.data, 'transcript.csv')
        lines = codecs.open(transcript, 'r', 'utf-8').readlines()
        for line in tqdm(lines, desc='load_data'):
            fields = line.strip().split("|")
            assert len(fields) >= 3
            fname, unnorm_text, norm_text = fields[:3]
            if len(fields) >= 4:
                phones = fields[4]
            if len(fields) >= 5:
                durations = fields[5]

            fpath = os.path.join(hp.data, "wav_norm", fname + ".wav")
            fpaths.append(fpath)

            if hp.input_type == 'phones':
                phones = phones_normalize(phones, char2idx) # in case of phones, all EOS markers are assumed included
                phones = [char2idx[char] for char in phones]
                text_lengths.append(len(phones))
                texts.append(np.array(phones, np.int32).tostring())
            elif hp.input_type == 'letters':
                text = text_normalize(norm_text, hp) + "E"  # E: EOS
                text = [char2idx[char] for char in text]
                text_lengths.append(len(text))
                texts.append(np.array(text, np.int32).tostring())                    
             
        if hp.n_utts > 0:
            assert hp.n_utts <= len(fpaths)
            return fpaths[:hp.n_utts], text_lengths[:hp.n_utts], texts[:hp.n_utts]
        else:    
            return fpaths, text_lengths, texts

    else: # synthesize on unseen test text.
        if "nancy" in hp.data:   ## get phones from heldout transcripts
            fpaths, text_lengths, texts = [], [], []
            transcript = os.path.join(hp.data, 'transcript.csv.test')
            lines = codecs.open(transcript, 'r', 'utf-8').readlines()[:3] ## TODO configure n synth 
            texts = np.zeros((len(lines), hp.max_N), np.int32)
            for (i, line) in enumerate(lines):
                fname, _, sent, phones, durations = line.strip().split("|")
                if hp.input_type == 'phones':                
                    phones = phones_normalize(phones, char2idx) + [hp.EOS]
                    # print (phones)
                    phones = [char2idx[char] for char in phones]
                    texts[i, :len(phones)] = phones
                elif hp.input_type == 'letters':
                    print (sent)
                    sent = text_normalize(sent, hp) + 'E'
                    #sent = text_normalize(sent, hp) + 'E'   #### this e lowered ! 2nd added  below!!!!!
					
                    print (sent)
                    ixx = [char2idx[char] for char in sent]
                    print (ixx)

                    texts[i, :len(sent)+1] = [char2idx[char] for char in text_normalize(sent, hp) + 'E' ]                    
                    #texts[i, :len(sent)+1] = [char2idx[char] for char in sent] # text_normalize(sent, hp) + 'E' ]    
					                
            return texts
        else:
            # Parse
            lines = codecs.open(hp.test_data, 'r', 'utf-8').readlines()[1:]
            sents = [text_normalize(line.split(" ", 1)[-1], hp).strip() + "E" for line in lines] # text normalization, E: EOS
            texts = np.zeros((len(sents), hp.max_N), np.int32)
            for i, sent in enumerate(sents):
                texts[i, :len(sent)] = [char2idx[char] for char in sent]
            return texts

def get_batch(hp):
    """Loads training data and put them in queues"""
    # print ('get_batch')
    with tf.device('/cpu:0'):
        # Load data
        fpaths, text_lengths, texts = load_data(hp) # list

        maxlen, minlen = max(text_lengths), min(text_lengths)

        # Calc total batch count
        num_batch = len(fpaths) // hp.B

        # Create Queues
        fpath, text_length, text = tf.train.slice_input_producer([fpaths, text_lengths, texts], shuffle=True)

        # Parse
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
        mel.set_shape((None, hp.n_mels))
        mag.set_shape((None, hp.full_dim))
        #mag.set_shape((None, hp.n_fft//2+1))  ### OSW: softcoded this

        # Batching
        _, (texts, mels, mags, fnames) = tf.contrib.training.bucket_by_sequence_length(
                                            input_length=text_length,
                                            tensors=[text, mel, mag, fname],
                                            batch_size=hp.B,
                                            bucket_boundaries=[i for i in range(minlen + 1, maxlen - 1, 20)],
                                            num_threads=8,
                                            capacity=hp.B*4,
                                            dynamic_pad=True)

    return texts, mels, mags, fnames, num_batch

