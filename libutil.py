

import os
import re
import numpy as np
import codecs
import imp



    
def safe_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
def writelist(seq, fname):
    path, _ = os.path.split(fname)
    safe_makedir(path)
    f = codecs.open(fname, 'w', encoding='utf8')
    f.write('\n'.join(seq) + '\n')
    f.close()
    
    
def readlist(fname):
    f = codecs.open(fname, 'r', encoding='utf8')
    data = f.readlines()
    f.close()
    data = [line.strip('\n') for line in data]
    data = [l for l in data if l != '']
    return data
    
def read_norm_data(fname, stream_names):
    out = {}
    vals = np.loadtxt(fname)
    mean_ix = 0
    for stream in stream_names:
        std_ix = mean_ix + 1
        out[stream] = (vals[mean_ix], vals[std_ix])
        mean_ix += 2
    return out
    

def makedirecs(direcs):
    for direc in direcs:
        if not os.path.isdir(direc):
            os.makedirs(direc)

def basename(fname):
    path, name = os.path.split(fname)
    base = re.sub('\.[^\.]+\Z','',name)
    return base    

get_basename = basename # alias
def get_speech(infile, dimension):
    f = open(infile, 'rb')
    speech = np.fromfile(f, dtype=np.float32)
    f.close()
    assert speech.size % float(dimension) == 0.0,'specified dimension %s not compatible with data'%(dimension)
    speech = speech.reshape((-1, dimension))
    return speech    

def put_speech(m_data, filename):
    m_data = np.array(m_data, 'float32') # Ensuring float32 output
    fid = open(filename, 'wb')
    m_data.tofile(fid)
    fid.close()
    return    

def save_floats_as_8bit(data, fname):
    '''
    Lossily store data in range [0, 1] with 8 bit resolution
    '''
    assert (data.max() <= 1.0) and (data.min() >= 0.0), (data.min(), data.max())

    maxval = np.iinfo(np.uint8).max
    data_scaled = (data * maxval).astype(np.uint8)
    np.save(fname, data_scaled)

def read_floats_from_8bit(fname):
    maxval = np.iinfo(np.uint8).max
    data = (np.load(fname).astype(np.float32)) / maxval
    assert (data.max() <= 1.0) and (data.min() >= 0.0), (data.min(), data.max())
    return data


def listconf(config):
    for thing in dir(config):
        print (thing, getattr(config, thing))

def load_config(config_fname):        
    config = os.path.abspath(config_fname)
    assert os.path.isfile(config)
    hp = imp.load_source('config', config)
    # hp = conf_mod.Hyperparams()
    # hp.config_name = basename(config_fname) ## string name of configuration
    # hp = finalise_config(hp)
    

    # listconf(hp)
    # sys.exit('--==ooOOOoo==---')
    
    return hp


# def finalise_config(hp):
#     '''
#     Add some derived values -- important to do this last in case only one
#     of a pair of contributing values is altered by a config.
#     '''
#     hp.full_dim = hp.n_fft//2+1
#     hp.hop_length = int(hp.sr * hp.frame_shift)
#     hp.win_length = int(hp.sr * hp.frame_length)    
#     hp.prepro = True  # we will never extract spectrograms on the fly
      
#     #### Set some paths:-
#     if not hp.topworkdir:
#         topdir = os.path.realpath(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
#         hp.topworkdir = os.path.join(topdir, 'work')

#     hp.voicedir = os.path.join(hp.topworkdir, hp.config_name)

#     hp.logdir = os.path.join(hp.voicedir, 'train')
#     hp.sampledir = os.path.join(hp.voicedir, 'synth')

#     if not hp.featuredir:
#         hp.featuredir = os.path.join(hp.voicedir, 'data')
#     hp.coarse_audio_dir = os.path.join(hp.featuredir, 'mels')
#     hp.full_mel_dir = os.path.join(hp.featuredir, 'mels_full')
#     hp.full_audio_dir = os.path.join(hp.featuredir, 'mags')
#     hp.attention_guide_dir = os.path.join(hp.featuredir, 'att')





## Snickery etc.:-
# def load_config(config_fname):
#     config = {}
#     execfile(config_fname, config)
#     del config['__builtins__']
#     _, config_name = os.path.split(config_fname)
#     config_name = config_name.replace('.cfg','').replace('.conf','')
#     config['config_name'] = config_name
#     return config    