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
See section 'Preparing a new database' below to learn how this data was made.

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






## Preparing a new database

This describes how the features at `/group/project/cstr2/owatts/data/` were made and will be useful for preparing new databases. 


### Download LJSpeech

```
DATADIR=/group/project/cstr2/owatts/data/  

cd $DATADIR
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bunzip2  LJSpeech-1.1.tar.bz2
tar xvf LJSpeech-1.1.tar
rm LJSpeech-1.1.tar
cd LJSpeech-1.1/

DATADIR=$DATADIR/LJSpeech-1.1
```


### Phonetise the transcription

The downloaded data contains a file called `metadata.csv` providing a transcription of the audio in plain text. Use Festival (with the combilex RPX lexicon installed) to phonetise this transcription:


```
cd $CODEDIR
python ./script/festival/csv2scm.py -i $DATADIR/LJSpeech-1.1/metadata.csv -o $DATADIR/LJSpeech-1.1/utts.data

cd $DATADIR/
FEST=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/tool/festival/festival/src/main/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript.csv
```

The resulting transcript contains lines like:

```
LJ001-0008||has never been surpassed.|<_START_> h a z <> n E v @ r <> b i n <> s @ p A s t <.> <_END_>
```

Things to note:

- A fourth field has been added for the phonetic transcript; the second field (for unnormalised text) has been left empty.
 
- The phonetic transcript contains phones enriched with punctuation, word boundaries, and utterance end markers (wrapped in `<...>`).

- The transcription is RP accent, which is not an ideal fit for LJSpeech (a US speaker). However, using a single phoneset means porting models between accents is more straightforward, and finetuning with DCTTS seems to handle the mismatch reasonably. To make the transcription a little more general, no postlexical rules are used by the Festival script meaning e.g. that the final r in "never" in the line above remains. The DCTTS model itself will have to learn that this segment is deleted in SE British speech. 

- The script `fix_transcript.py` is needed to tidy up many spurious "punctuation marks" left due to inadequacies of the .scm script which occur due to how the 'next token' is found in hyphenated words and initialisms. One thing that is not resolved by postprocessing is "'s" in cases where it is (often wrongly) parsed as abbreviated 'is' or 'has' rather than possessive; in these cases it is represented in the phonetic transcript as `<'s>` rather than as an actual phone or phones. Again, it is up to DCTTS to learn whether this should be realised as [s], [z] or [Iz] depending on context.

- (In fact, note that word-internal r's are already removed like in ` l @@ n` -- need to do something more principled with multiple accents -- discuss with JT.)

- We aim to replace the use of Festival for phonetisation in the near future.  






### Get a list of phones used

Configuration needs to contain a list of phones used. To print this - and various other information about the corpus - to the terminal use the following script:

```
cd $CODEDIR
python ./script/check_transcript.py -i $DATADIR/transcript.csv -phone
````


Paste the Python list 'Observed phones:' into config as e.g. phonelist, so that variable `vocab` is defined as follows:
```
vocab = ['<PADDING>'] + phonelist
```

(i.e. with some non-phone symbol at index 0)








### Preprocess waveforms 

Normalise level and trim end silences based only on acoustics:

```
cd $CODEDIR

python ./script/normalise_level.py -i $DATADIR/wavs -o $DATADIR/wav_norm/ -ncores 25

./util/submit_tf_cpu.sh ./script/split_speech.py -w $DATADIR/wav_norm/ -o $DATADIR/wav_trim/ -dB 30 -ncores 25 -trimonly
```

Despite its name, `split_speech.py` only trims end silences when used with the `-trimonly` flag. It is worth listening to a few trimmed waveforms to check the level used (30 dB) is appropriate. It works for LJSpeech, but might need adjusting for other databases. Reduce this value to trim more aggressively.

All steps with `-ncores` flag can use multiple CPUs in parallel: please be considerate to other users of machines when using this flag.

Clean up by removing untrimmed and unnormalised data: 
```
rm -r $DATADIR/wavs  $DATADIR/wav_norm/
```









### Configure lengths of data


After running `prepare_acoustic_features.py`, inspect dataset using `check_transcript.py` again with the extra `-cmp` flag:

```
python ./script/check_transcript.py -i /disk/scratch/oliver/data/LJSpeech-1.1/transcript.csv -phone -cmp work/lj_01/data/mels
```

Need to set `max_N` in config file to value at least as big as most letters in a sentence, and `max_T` at least as big as most (coarse) frames in the audio of a sentence. At the same time, some datasets will have a large tail of a few long utterances, and this will make training less efficient. Look at the histograms printed by the script to terminal to decide a good cutoff point. LJ dataset is nice in that there is no long tail:


```

Letter/phone length max:
164
Frame length max:
202
    --------------------------------------------------------------
    |          Histogram of sentence length in letters           |
    --------------------------------------------------------------

 829|                o
 737|                ooooooo
 645|             oooooooooo
 553|            oooooooooooo
 461|          ooooooooooooooo
 369|         oooooooooooooooo
 277|       ooooooooooooooooooo
 185|     oooooooooooooooooooooo
  93|    oooooooooooooooooooooooo
   2| oooooooooooooooooooooooooooooo
     ------------------------------
     1 2 3 4 5 6 7 8 9 1 1 1 1 1 1
     2 2 2 2 2 2 2 2 3 0 1 2 3 4 5
       . . . . . . . . 3 3 3 3 3 3
       1 2 4 5 6 8 9 0 . . . . . .
       3 6 0 3 6 0 3 6 2 3 4 6 7 8
       3 6 0 3 6 0 3 6 0 3 6 0 3 6
    ----------------------------------------------------------------
    |            Histogram of sentence length in frames            |
    ----------------------------------------------------------------

 573|                  oo  ooooooo
 510|                ooooooooooooooo
 447|             o ooooooooooooooooo
 384|             ooooooooooooooooooo
 321|          o oooooooooooooooooooo
 258|        oooooooooooooooooooooooo
 195|      oooooooooooooooooooooooooo
 132|     ooooooooooooooooooooooooooo
  69|   ooooooooooooooooooooooooooooo
   6| ooooooooooooooooooooooooooooooo
     -------------------------------
     2 3 4 5 6 8 9 1 1 1 1 1 1 1 1 2
     1 3 5 7 9 1 3 0 1 2 4 5 6 7 8 0
       . . . . . . 5 7 9 1 3 5 7 9 1
       0 1 1 2 3 3 . . . . . . . . .
       6 3 9 6 3 9 4 5 5 6 7 7 8 9 9
```



... so we'll configure max_N and max_T to be the actual maxima (164, 202):

```
    max_N = 164 # Maximum number of characters.
    max_T = 202 # Maximum number of mel frames.
```

Note that if longer sentences are in dataset, they will be filtered and discarded when data is loaded before training.




## Multispeaker system

```
  cd /group/project/cstr2/owatts/data/
  530  ls
  531  mv VCTK-Corpus/ VCTK-Corpus_OLD
  532  wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip
   unzip VCTK-Corpus.zip
cd VCTK-Corpus

rm -f metadata.csv
ls txt/*/*.txt | while read TXT ; do 
      BASE=`basename $TXT .txt`
      LINE=`cat $TXT` ; 
      SPKR=`echo $BASE | awk -F_ '{print $1}'`
      if [ $SPKR == p376 ] ; then
         LINE=${LINE:1:-1} ; ## remove initial and final " in p376 data
      fi 
      echo "$BASE||$LINE"; 
done >> metadata.csv



CODEDIR=/disk/scratch/oliver/dc_tts_osw_clean
DATADIR=/group/project/cstr2/owatts/data/VCTK-Corpus
cd $CODEDIR
python ./script/festival/csv2scm.py -i $DATADIR/metadata.csv -o $DATADIR/utts.data

cd $DATADIR/
FEST=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/tool/festival/festival/src/main/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript_temp2.csv


# add speaker codes in last field

awk -F_ '{print $1}' ./transcript_temp2.csv > speakers.tmp
paste -d\| ./transcript_temp2.csv speakers.tmp > ./transcript.csv


# put all waves in same dir (would prob break on AFS?)

mkdir wav
mv wav48/p*/*.wav ./wav/


cd $CODEDIR

python ./script/normalise_level.py -i $DATADIR/wav -o $DATADIR/wav_norm/ -ncores 25

rm -r  $DATADIR/wav


## trim with split wave version 29ef9253253f7c63e4fbebaf06e4e70709c70d68

./util/submit_tf_cpu.sh ./script/split_speech.py -w $DATADIR/wav_norm/ -o $DATADIR/wav_trim_15dB/ -dB 15 -ncores 25 -trimonly


#############################


cd $CODEDIR
./util/submit_tf_cpu.sh ./prepare_acoustic_features.py -c ./config/vctk_01.cfg -ncores 25
./util/submit_tf.sh ./prepare_attention_guides.py -c ./config/vctk_01.cfg -ncores 25


python ./script/check_transcript.py -i $DATADIR/transcript.csv -cmp work/vctk_01/data/mels/ -phone

python ./script/check_transcript.py -i $DATADIR/transcript.csv -cmp work/vctk_01/data/mels/ -phone  -maxframes 100 -maxletters 80



```