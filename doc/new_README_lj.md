# Ophelia

A modified version of Kyubyong Park's [dc_tts repository](https://github.com/Kyubyong/dc_tts), which implements a variant of the system described in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).

## Tools


Go to a suitable location and clone repository:

```
git clone https://github.com/oliverwatts/ophelia.git
cd ophelia
CODEDIR=`pwd`
```

## Installation of Python dependencies with virtual environment


Make a directory to house virtual environments if you don't already have one, and move to it:

```
mkdir /convenient/location/virtual_python/
cd /convenient/location/virtual_python/
virtualenv --distribute --python=/usr/bin/python2.7 venv_dctts
source /convenient/location/virtual_python/venv_dctts/bin/activate
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

## Data


We will use the LJ speech dataset, this is ~24 hrs audiobook read by a US female speaker. To download it and extract the contents, run:

```
DATADIR=/some/convenient/directory
wget https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
tar -xvf LJSpeech-1.1.tar.bz2
DATADIR=$DATADIR/LJSpeech-1.1
```

For more details on the dataset, visit the webpage: https://keithito.com/LJ-Speech-Dataset/

## Data preparation (1): installing Festival and obtaining phonetic transcriptions


The downloaded data contains a file called `metadata.csv` providing a transcription of the audio in plain text. Use Festival with the CMU lexicon to phonetise this transcription.

If you don't have a Festival installation, you can obtain one by running:

```
INSTALL_DIR=/some/convenient/directory/festival

mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festival-2.4-release.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz

tar xvf festival-2.4-release.tar.gz
tar xvf speech_tools-2.4-release.tar.gz

## Install Speech tools first
cd speech_tools
./configure  --prefix=$INSTALL_DIR
gmake

## Then compile Festival
cd ../festival
./configure  --prefix=$INSTALL_DIR
gmake

# Finally, get a Festival voice with the CMU lexicon
cd ..
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_cmu_us_awb_cg.tar.gz
tar xvf festvox_cmu_us_awb_cg.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_CMU.tar.gz
tar xvf festlex_CMU.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_POSLEX.tar.gz
tar xvf festlex_POSLEX.tar.gz
```

To test the installation, open Festival and load the voice.
Run the *locally installed* festival (NB: initial ./ is important!)

```
./festival/bin/festival
festival> (voice_cmu_us_awb_cg)
festival> (SayText "If i'm speaking then installation actually went ok.")
festival> (quit)
```

Now, to phonetise the LJ transcription, you will pass the `metadata.csv` file through Festival and obtain phone transcriptions with the CMU lexicon.

```
cd $CODEDIR
# Get a file formatting the sentences in the right way for Festival, the "utts.data" file
python ./script/festival/csv2scm.py -i $DATADIR/metadata.csv -o $DATADIR/utts.data

cd $DATADIR/
FEST=$INSTALL_DIR
SCRIPT=$CODEDIR/script/festival/make_rich_phones_cmulex.scm
$FEST/festival/bin/festival -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript.csv
```

During the process you should see the print of the resulting transcription, for example:

```
LJ003-0043||and it was not ready to relieve Newgate till late in eighteen fifteen.|<_START_> ae n d <> ih t <> w aa z <> n aa t <> r eh d iy <> t ax <> r ih l iy v <> n uw g ey t <> t ih l <> l ey t <> ih n <> ey t iy n <> f ih f t iy n <.> <_END_>
```

You can see that each line in `transcript.csv` contains four fields separated by the pipe (|) symbol. The first one in the name of the wav file. The second one corresponds to unnormalised text (in this case empty). The third field is the normalised text. The fourth field contains the phonetic transcription, enriched with: starting and ending symbols, word boundaries, and special punctuation symbols.

## Data preparation (2): pre-process waveforms


Normalise level and trim end silences based only on acoustics:

```
cd $CODEDIR

python ./script/normalise_level.py -i $DATADIR/wavs -o $DATADIR/wav_norm/ -ncores 25

./util/submit_tf_cpu.sh ./script/split_speech.py -w $DATADIR/wav_norm/ -o $DATADIR/wav_trim/ -dB 30 -ncores 25 -trimonly
```

Despite its name, `split_speech.py` only trims end silences when used with the `-trimonly` flag. It is worth listening to a few trimmed waveforms to check the level used (30 dB) is appropriate. It works for LJSpeech, but might need adjusting for other databases. Reduce this value to trim more aggressively.


Clean up by removing untrimmed and unnormalised data:
```
rm -r $DATADIR/wavs  $DATADIR/wav_norm/
```

## The config file: path to waveforms

Build a new config file for your project, by making your own copy of `config/lj_tutorial.cfg`.
You will have to modify the value `datadir`, by adding the path to the LJ folder.

```
# Modify in config file
datadir = '/path/to/LJSpeech-1.1/'
```


## Extract acoustic features


Use the config file to extract acoustic features. The acoustic features are mels and mags. You only need to run this once per dataset.


```
cd $CODEDIR
./util/submit_tf_cpu.sh ./prepare_acoustic_features.py -c ./config/lj_tutorial.cfg -ncores 25
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


## The config file: get length of N and T


We need to provide to the config file the maximum length of the phone transcriptions and the coarse mels (the inputs and outputs to the T2M model).

```
cd $CODEDIR
python $CODEDIR/script/check_transcript.py -i $DATADIR/transcript.csv -phone -cmp $CODEDIR/work/lj_tutorial/data/mels

```

The output should look like this. The script is giving information about the length of the sequences, the phone set in the transcriptions, and a histogram of the lenghts. In the config file, you can use the maximum length, or you can choose a different cutting point, for example, if you only have one sentence at that max length but most of your data is below that range.

```
------------------
Observed phones:
------------------

['<!>', '<">', "<'>", "<'s>", '<)>', '<,>', '<.>', '<:>', '<;>', '<>', '<?>', '<]>', '<_END_>', '<_START_>', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
------------------

Letter/phone length max:
166

```

We will add these to the config file. You can see in there in `vocab` there is the list of phones. Change the max_N and max_T to the values given by the script.


```
# In the config file
max_N = 150 # Maximum number of characters/phones
max_T = 300 # Maximum number of mel frames

```


### Make per-utterance attention guides

The configuration file allows for two options for guided attention. If you leave an empty string for the variable `attention_guide_dir`, global attention matrix will be used, of size `(max_N, max_T)`. Otherwise, if there is a path given, then attention guides per utterance length will be constructed.

```
# In the config file
attention_guide_dir = ''
```

Otherwise, if there is a path given, then attention guides per utterance length will be constructed. Run:

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
