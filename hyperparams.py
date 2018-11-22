# -*- coding: utf-8 -*-
#/usr/bin/python2
'''
By kyubyong park. kbpark.linguist@gmail.com. 
https://www.github.com/kyubyong/dc_tts
'''

config_name = 'DUMMY' ## this will be provided in modules which make configs as 
                      ## subclass from DefaultHyperparams

class DefaultHyperparams:
    '''Hyper parameters'''

    topworkdir = ''

    input_type = 'letters' ## letters or phones

    # pipeline
    prepro = True  # if True, run `python prepro.py` first before running `python train.py`.
    
    normtype = 'minmax'
    trim_before_spectrogram_extraction = 0
    extract_full_mel = False  ### extract mel at normal frame rate?
    bucket_by = 'textlength'

    # signal processing
    vocoder = 'griffin_lim'  
    sr = 22050  # Sampling rate.
    n_fft = 2048  # fft points (samples)
    frame_shift = 0.0125  # seconds
    frame_length = 0.05  # seconds
    hop_length = int(sr * frame_shift)  # samples. =276.
    win_length = int(sr * frame_length)  # samples. =1102.
    n_mels = 80  # Number of Mel banks to generate
    full_dim = n_fft//2+1
    power = 1.5  # Exponent for amplifying the predicted magnitude
    n_iter = 50  # Number of inversion iterations
    preemphasis = .97
    max_db = 100
    ref_db = 20

    
    coarse_audio_dir = topworkdir + '/mels/'
    full_audio_dir =  topworkdir + '/mags/'
    norm_stats_file = ''

    use_bd1_loss = True
    use_bd2_loss = True
    


    # Model
    r = 4 # Reduction factor. Do not change this.
    dropout_rate = 0.05
    e = 128 # == embedding
    d = 256 # == hidden units of Text2Mel
    c = full_dim / 2 # 512 # == hidden units of SSRN
    attention_win_size = 3
    norm = 'layer' ## type of normalisation layers to use: form ['layer', 'batch', None]



    # data
    transcript = '/disk/scratch/oliver/data/nancy/transcript.csv'
    waveforms = '/disk/scratch/oliver/data/nancy/wav_norm'
    n_utts = 0 ## 0 means use all data, other positive integer means select this many sentences
    test_data = 'test_osws.txt'
    vocab = "PE abcdefghijklmnopqrstuvwxyz'.?" # P: Padding, E: EOS.
    max_N = 180 # Maximum number of characters.
    max_T = 210 # Maximum number of mel frames.




    # training scheme
    restart_from_savepath = ''
    lr = 0.001 # Initial learning rate.
    B = 32 # batch size
    save_every_n_iterations = 1000
    num_iterations = 2000000
