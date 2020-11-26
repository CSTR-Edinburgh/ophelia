# Ophelia

A modified version of Kyubyong Park's [dc_tts repository](https://github.com/Kyubyong/dc_tts), which implements a variant of the system described in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969).

## Recent additions

- turn on/off monotonic attention during synthesis  
- other attention-based losses (CDP, Ain and Aout), see "Confidence through attention", http://arxiv.org/abs/1710.03743  
- MerlinTextEncoder (TextEncoder that takes Merlin labels as input)  

## Tools


Go to a suitable location and clone repository:

```
git clone https://github.com/CSTR-Edinburgh/ophelia.git
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

## Data preparation (1): installing Festival


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

## Data preparation (2): obtaining phonetic transcriptions

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

## Data preparation (3): validation and test sets

From the `transcript.csv` we will take the last 10 sentences to build a `test_transcript.csv`. The rest of the chapter 50 will be used as validation data (you can change this in the config file).

```
tail -n 10 transcript.csv > test_transcript.csv

```

## Data preparation (4): pre-process waveforms


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
./util/submit_tf_cpu.sh $CODEDIR/prepare_acoustic_features.py -c $CODEDIR/config/lj_tutorial.cfg -ncores 25
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

Some of the output should look like this. The script is giving information about the length of the sequences, the phone set in the transcriptions, and a histogram of the lengths. In the config file, you can use the maximum length, or you can choose a different cutting point, for example, if you only have one sentence at that max length but most of your data is below that value.

```
------------------
Observed phones:
------------------

['<!>', '<">', "<'>", "<'s>", '<)>', '<,>', '<.>', '<:>', '<;>', '<>', '<?>', '<]>', '<_END_>', '<_START_>', 'aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ay', 'b', 'ch', 'd', 'dh', 'eh', 'er', 'ey', 'f', 'g', 'hh', 'ih', 'iy', 'jh', 'k', 'l', 'm', 'n', 'ng', 'ow', 'oy', 'p', 'r', 's', 'sh', 't', 'th', 'uh', 'uw', 'v', 'w', 'y', 'z', 'zh']
------------------

Letter/phone length max:
166
Frame length max:
203

```

We will add these to the config file. You can see in there in `vocab` there is the list of phones. Change the max_N and max_T to the values given by the script.


```
# In the config file
max_N = 150 # Maximum number of characters/phones
max_T = 200 # Maximum number of mel frames

```


### Make per-utterance attention guides

The configuration file allows for two options for guided attention. If you leave an empty string for the variable `attention_guide_dir`, global attention matrix will be used, of size `(max_N, max_T)`. Otherwise, if there is a path given, then attention guides per utterance length will be constructed.

```
# In the config file
attention_guide_dir = ''
```

Otherwise, you can add a path or use the one already there, and then build attention guides per utterance length. Run:

```
$CODEDIR/util/submit_tf.sh $CODEDIR/prepare_attention_guides.py -c $CODEDIR/config/lj_tutorial.cfg -ncores 25
```



## Train Text2Mel and SSRN networks:

The config `lj_tutorial` trains on only a few sentences for a limited number of epochs. It
won't produce anything decent-sounding, but use it to check the tools work. You have to train separately the T2M model from the SSRN. If you have multiple GPUs, you can train both in parallel.  


```
$CODEDIR/util/submit_tf.sh $CODEDIR/train.py -c $CODEDIR/config/lj_tutorial.cfg -m t2m
$CODEDIR/util/submit_tf.sh $CODEDIR/train.py -c $CODEDIR/config/lj_tutorial.cfg -m ssrn
```

Note the use of `./util/submit_tf.sh`, which will reserve a single GPU and make only that one visible for the job you are starting, and add some necessary resources to system path.

If you want to train the whole dataset, you have to change in the config file `n_utts = 200`, to `n_utts = 0`, as well as the number of epochs in `max_epochs = 10`. If you get any errors regarding memory, you can decrease the batch size in `batchsize = {'t2m': 32, 'ssrn': 8}` (as mels are longer, you might need to decrease especially the ssrn batch size).


## Synthesise:

Use the last saved model to synthesise N sentences from the test set:

```
$CODEDIR/util/submit_tf.sh $CODEDIR/synthesize.py -c $CODEDIR/config/lj_tutorial.cfg -N 5
```

As promised, this will not sound at all like speech.

## Synthesise validation data from many points in training


In the config file you can see the parameter `validate_every_n_epochs = 1`, this means the model will generate parameters that will be saved in separate folders in `work/<config-name>`. As objective evaluations are not as reliable as listening to the samples, you can synthesize these parameters to have an idea of the sound that the model is able to generate at the given epoch.

```
$CODEDIR/util/submit_tf.sh $CODEDIR/synthesise_validation_waveforms.py -c $CODEDIR/config/lj_tutorial.cfg -ncores 25
```


## Other recipes

[Training LJ with letter input](./doc/recipe_lj_letters.md)

[Multispeaker training](./doc/recipe_vctk.md)

[Nancy](./doc/recipe_nancy.md)

[Adapt Nancy to Nick](./doc/recipe_nancy2nick.md)
