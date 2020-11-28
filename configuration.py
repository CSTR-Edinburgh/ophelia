# -*- coding: utf-8 -*-
#!/usr/bin/env python2
import os
import imp
import inspect


CONFIG_DEFAULTS = [
    ('initialise_weights_from_existing', [], ''),
    ('update_weights', [], ''),
    ('num_threads', 8, 'how many threads get_batch should use to build training batches of data (default: 8)'),
    ('plot_attention_every_n_epochs', 0, 'set to 0 if you do not wish to plot attention matrices'),
    ('num_sentences_to_plot_attention', 0, 'number of sentences to plot attention matrices for'),
    ('concatenate_query', True, 'Concatenate [R Q] to get audio decoder input, or just take R?'),
    ('use_external_durations', False, 'Use externally supplied durations in 6th field of transcript for fixed attention matrix A'),
    ('text_encoder_type', 'DCTTS_standard', 'one of DCTTS_standard/none/minimal_feedforward'),
    ('merlin_label_dir', '', 'npy format phone labels converted from merlin using process_merlin_label.py'),
    ('merlin_lab_dim', 592, ''),
    ('bucket_data_by', 'text_length', 'One of audio_length/text_length. Label length will be used if merlin_label_dir is set and bucket_data_by=="text_length"'),
    ('history_type', 'DCTTS_standard', 'DCTTS_standard/fractional_position_in_phone/absolute_position_in_phone/minimal_history'),
    ('beta1', 0.9, 'ADAM setting - default value from original dctss repo'),
    ('beta2', 0.999, 'ADAM setting - default value from original dctss repo'),
    ('epsilon', 0.00000001 , 'ADAM setting - default value from original dctss repo'),
    ('decay_lr', True , 'learning rate decay - default value from original dctss repo'),
    ('squash_output_t2m', True, 'apply sigmoid to output - binary divergence loss will be disabled if False'),
    ('squash_output_ssrn', True, 'apply sigmoid to output - binary divergence loss will be disabled if False'),
    ('store_synth_features', False, 'store .npy file of features alongside output .wav file'),
    ('turn_off_monotonic_for_synthesis',False, 'turns off FIA mechanism for synthesis, should be False during training'),
    ('lw_cdp',0.0,''),
    ('lw_ain',0.0,''),
    ('lw_aout',0.0,''),
    ('attention_guide_fa',False,'use attention guide as target - MSE attention loss'),
    ('select_central',False,'use only centre phones from Merlin labels'),
    ('MerlinTextEncWithPhoneEmbedding',False,'use Merlin labels and phone embeddings as input of TextEncoder')
]

## Intended to have hp as a module, but this doesn't allow pickling and therefore 
## use in parallel processing. So, convert module into an object of arbitrary type 
## ("Hyperparams") having same attributes: 
class Hyperparams(object):
    def __init__(self, module_object):
        for (key, value) in module_object.__dict__.items():
            if key.startswith('_'):
                continue
            if inspect.ismodule(value): # e.g. from os imported at top of config
                continue
            setattr(self, key, module_object.__dict__[key])
    def validate(self):
        '''
        Supply defaults for various thing of appropriate type if missing -- 
        TODO: Currently this is just to supply values for variables added later in development.
        Should we have some filling in of defaults like this more permanently, or should
        everything be explicitly set in a config file?
        '''
        for (varname, default_value, help_string) in CONFIG_DEFAULTS:
            if not hasattr(self, varname):
                setattr(self, varname, default_value)


def load_config(config_fname):        
    config = os.path.abspath(config_fname)
    assert os.path.isfile(config), 'Config file %s does not exist'%(config)
    settings = imp.load_source('config', config)
    hp = Hyperparams(settings)
    hp.validate()
    return hp



### https://stackoverflow.com/questions/1325673/how-to-add-property-to-a-class-dynamically

# class atdict(dict):
#     __getattr__= dict.__getitem__
#     __setattr__= dict.__setitem__
#     __delattr__= dict.__delitem__
