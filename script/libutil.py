import os
import re
import numpy as np

#### TODO -- this module is duplicated 1 level up -- sort this out!


def load_config(config_fname):
    config = {}
    execfile(config_fname, config)
    del config['__builtins__']
    _, config_name = os.path.split(config_fname)
    config_name = config_name.replace('.cfg','').replace('.conf','')
    config['config_name'] = config_name
    return config

    
def safe_makedir(dir):
    if not os.path.isdir(dir):
        os.makedirs(dir)
        
def writelist(seq, fname):
    f = open(fname, 'w')
    f.write('\n'.join(seq) + '\n')
    f.close()
    
def readlist(fname):
    f = open(fname, 'r')
    data = f.readlines()
    f.close()
    return [line.strip('\n') for line in data]
    
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