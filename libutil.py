

import os
import re
import numpy as np
import codecs
import imp
import inspect


    
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



### https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically


# class atdict(dict):
#     __getattr__= dict.__getitem__
#     __setattr__= dict.__setitem__
#     __delattr__= dict.__delitem__



## Intended to have hp as a module, but this doesn't allow pickling and therefore 
## use in parallel processing. So, convert it into an object with same attributes: 
class Hyperparams(object):
    def __init__(self, module_object):
        for (key, value) in module_object.__dict__.items():
            if key.startswith('_'):
                continue
            if inspect.ismodule(value): # e.g. from os imported at top of config
                continue
            #print (key, value)
            setattr(self, key, module_object.__dict__[key])
     

def load_config(config_fname):        
    config = os.path.abspath(config_fname)
    assert os.path.isfile(config)
    settings = imp.load_source('config', config)
    hp = Hyperparams(settings)
    return hp


## Snickery etc.:-
def load_config2(config_fname):
    config = {}
    execfile(config_fname, config)
    del config['__builtins__']
    #_, config_name = os.path.split(config_fname)
    #config_name = config_name.replace('.cfg','').replace('.conf','')
    #config['config_name'] = config_name
    # class atdict(dict):
    #     __getattr__= dict.__getitem__
    #     __setattr__= dict.__setitem__
    #     __delattr__= dict.__delitem__    
    return config  

