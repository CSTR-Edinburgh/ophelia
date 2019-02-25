# dc_tts_osw

A modified version of Kyubyong Park's [dc_tts repository](https://github.com/Kyubyong/dc_tts), which implements a variant of the system described in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969). 

TODO: note differences between the paper and Kyubyong Park's implementation

TODO: Note differences between this and Kyubyong Park's version

TODO: choose a better name than dc_tts_osw ;-)


## Tools

Go to a suitable location and clone repository:

```
git clone https://github.com/oliverwatts/dc_tts_osw.git
cd dc_tts_osw
CODEDIR=`pwd`
```



## Installation of Python dependencies with virtual environment


Make a directory to house virtual environments if you don't already have one, and move to it:

```
mkdir /convenient/location/virtual_python/
cd /convenient/location/virtual_python/
virtualenv --distribute --python=/usr/bin/python2.7 dctts
source /convenient/location/virtual_python/dctts/bin/activate
```

With the virtual environment activated, you can now install the necessary packages.

```
cd $CODEDIR
pip install --upgrade pip
```

Then:

```
pip install -r ./requirements.txt
```








## Extract acoustic features 

The first part of these notes assumes that you have already prepared waveforms and transcriptions. An example of such data can be found at `/group/project/cstr2/owatts/data/LJSpeech-1.1/` - please feel free to use this.
See [Preparing a new database](./doc/preparing_new_database.md) to learn how this data was made.

First extract some features from the existing data using the config file `./configs/lj_test.cfg`. (If you are using your own data in another location, change the relevant path in the config.) The argument `-ncores` can use multiple cores in parallel: please be considerate and check if others are using a shared machine before setting this very high. 

```
./util/submit_tf_cpu.sh ./prepare_acoustic_features.py -c ./config/lj_test.cfg -ncores 25
```

This will output data by default in directories under:

```
$CODEDIR/work/<config>/data/
```

This can take quite a bit of space:

```
du -sh work/lj_test/data/*
2.1G    work/lj_test/data/full_mels
27G     work/lj_test/data/mags
548M    work/lj_test/data/mels
```

... so alternatively if compatible acoustic features already exist, reuse them either by softlinking from the older features:

```
cp -rs $CODEDIR/work/<old_config_name>/data/ $CODEDIR/work/<new_config_name>/
```

... or configuring `featuredir` appropriately.

<!-- NB:  (random shift in reduction):
 -->




### Make per-utterance attention guides

Configuring:

```
attention_guide_dir = ''
```

... would use a single global attention guide as in Park's original code, of size `(max_N, max_T)`. This might be bad if there is much difference in rate between sentences, as would be the case in multispeaker databases. Use this command to prepare per-utterance guides with a config such as `ls_test.cfg` where `attention_guide_dir` points to a directory path (non-empty string): 

```
./util/submit_tf.sh ./prepare_attention_guides.py -c ./config/lj_test.cfg -ncores 25
```




## Train Text2Mel and SSRN networks (possibly simultaneously):

The config `lj_test` trains on only a few sentences for a limited number of epochs. It 
won't produce anything decent-sounding, but use it to check the tools work:


```
./util/submit_tf.sh ./train.py -c ./config/lj_test.cfg -m t2m
./util/submit_tf.sh ./train.py -c ./config/lj_test.cfg -m ssrn
```

Note the use of `./util/submit_tf.sh`, which will reserve a single GPU and make only that one visible for the job you are starting, and add some necessary resources to system path. 


## Synthesise:

Use the last saved model to synthesise 10 sentences from the test set:

```
./util/submit_tf.sh ./synthesize.py -c ./config/lj_test.cfg -N 10
```

Synthetic speech is output at `./work/lj_test/synth/4_4/*.wav`; adjust path `$DEST` inside
`util/sync_output_to_afs.sh` then use it to export the audio to somewhere more convenient:

```
./util/sync_output_to_afs.sh lj_test
```

As promised, this will not sound at all like speech.


## Synthesise validation data from many points in training

```
./util/submit_tf.sh synthesise_validation_waveforms.py -c config/lj_test.cfg -ncores 25
```



## Cleaning up

Each copy of the model parameters takes c.300MB of disk -- best system to removed unwanted intermediate models?

## Run on more data

Repeat the above with config `lj_01` to use the whole database. 

Note the following config values which determine for how long the model is trained:

```
batchsize = {'t2m': 32, 'ssrn': 32}
validate_every_n_epochs = 10   ## how often to compute validation score and save speech parameters
save_every_n_epochs = 20  ## as well as 5 latest models, how often to archive a model
max_epochs = 300
```

The most recent model is stored after each each epoch, and the 5 most recent such models are stored before being overwritten. 

(LJ data has 400 batches of 32 sentences, so that 1 epoch = 400 steps.)


## Utilities

The directory `./utils/` contains a few useful scripts. The `submit*` scripts 
have been used in the examples above. TODO: mention the sync scripts. 

<!-- The script `sync_code_from_afs.sh` 
is useful to keep code in a central, easier-to-edit place than a GPU server 
(e.g. in AFS space) in sync with the code you are running. 



gpu_lock.py           submit_tf_cpu.sh      sync_output_to_afs.sh
submit_tf.sh          sync_code_from_afs.sh

 -->



## Recipes

[Multispeaker training](./doc/recipe_vctk.md)

[Nancy](./doc/recipe_nancy.md)

[Adapt Nancy to Nick](./doc/recipe_nancy2nick.md)


