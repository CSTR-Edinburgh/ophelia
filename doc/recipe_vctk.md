# Multispeaker system with VCTK data

Note that the best way of training a multispeaker model is yet to be determined. This document will describe how to train some different multispeaker models that are possible in Ophelia. Festival will be used for the phonetic transcription ([installation](./festival_install.md)).


## Download the VCTK data and create a metadata.csv file
```
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/2651/VCTK-Corpus.zip
unzip VCTK-Corpus.zip
cd VCTK-Corpus

ls txt/*/*.txt | while read TXT ; do 
      BASE=`basename $TXT .txt`
      LINE=`cat $TXT` ; 
      SPKR=`echo $BASE | awk -F_ '{print $1}'`
      if [ $SPKR == p376 ] ; then
         LINE=${LINE:1:-1} ; ## remove initial and final " in p376 data
      fi 
      echo "$BASE||$LINE"; 
done >> metadata.csv
```

## Create an utts.data file as required by Festival

```
CODEDIR=/path/to/ophelia
DATADIR=/path/to/VCTK-Corpus
cd $CODEDIR
python ./script/festival/csv2scm.py -i $DATADIR/metadata.csv -o $DATADIR/utts.data
```

## Make a phonetic transcription

You need to be in the same directory as the utts.data file for this command to run (which should be in $DATADIR).

```
cd $DATADIR/
FEST=/path/to/your/installation/of/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript_temp2.csv
```

Note that this transcription will be using RP English, which obviously doesn't match all the VCTK speakers' accents. See [Prepare a new database](./preparing_new_database.md) for some notes on this.

Add speaker codes to the last fields in the transcript by either running:

```
awk -F_ '{print $1}' ./transcript_temp2.csv > speakers.tmp
paste -d\| ./transcript_temp2.csv speakers.tmp > ./transcript.csv
```

or:

```
python $CODEDIR/script/festival/multi_transcript.py -i ./transcript_temp2.csv -o ./transcript.csv
```

## Make a test transcript using Harvard sentences

```
mkdir $DATADIR/test_set
cd $DATADIR/test_set

ls /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/txt/hvd*.txt | while read TXT ; do 
      BASE=`basename $TXT .txt`
      LINE=`cat $TXT` ; 
      echo "$BASE||$LINE"; 
done >> harvard.csv

python ./script/festival/csv2scm.py -i $DATADIR/test_set/harvard.csv -o $DATADIR/test_set/utts.data

FEST=/path/to/your/installation/of/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./harvard_tmp.csv

python $CODEDIR/script/festival/fix_transcript.py ./harvard_tmp.csv > ./harvard_combilex_rpx.csv
```

## Prepare the data

The wav files should all be in one directory:

```
mkdir $DATADIR/wav
mv $DATADIR/wav48/p*/*.wav ./wav/
```

Normalise levels by running:

```
cd $CODEDIR

python ./script/normalise_level.py -i $DATADIR/wav -o $DATADIR/wav_norm/ -ncores 25

rm -r  $DATADIR/wav
```

The wav directory is removed to save spaces. Thereafter, trim silences from the data:

```
./util/submit_tf_cpu.sh ./script/split_speech.py -w $DATADIR/wav_norm/ -o $DATADIR/wav_trim_15dB/ -dB 15 -ncores 25 -trimonly
```

## Extract acoustic features and attention guides

The configuration file holds paths to the data, transcripts, etc., which should be changed to match the way your data and files are structured. Thereafter, these two commands can be run:

```
./util/submit_tf_cpu.sh ./prepare_acoustic_features.py -c ./config/vctk_01.cfg -ncores 25
./util/submit_tf.sh ./prepare_attention_guides.py -c ./config/vctk_01.cfg -ncores 25
```

## Edit the configuration file

The following command will output a list of phones. Paste this into the config file as 'vocab', adding '\<PADDING\>' as an entry to the list. It also produces a list of the speakers - these should be pasted into the config file as 'speaker_list', also adding '\<PADDING\>' as an entry to the list.

```
python ./script/check_transcript.py -i $DATADIR/transcript.csv -cmp work/vctk_01/data/mels/ -phone -speaker
```

The histograms that are also shown as outputs should be used to decide a good cutoff point if there is a large tail of long utterances. (Those utterances make training more inefficient. The script can be rerun with the arguments '-maxframes' and '-maxletters' while elaborating with good cutoff points. When having decided, set the variables 'max_N' and 'max_T' to the max number of characters and of coarse frames in a sentence, respectively.

The variable 'multispeaker' in the configuration file can also be changed - it specifies at what locations in the network speaker codes should be added. See the config file for possible ones. That argument can also be set to ['learn_channel_contrubutions'] for lcc.

## Training

Run the following two commands (possibly simultaneously):

```
./util/submit_tf.sh ./train.py -c config/vctk_01.cfg -m t2m
./util/submit_tf.sh ./train.py -c config/vctk_01.cfg -m ssrn
```

To train a new model, but using a previously trained SSRN, simply softlink the old one to the new model's 'work' directory:

```
mkdir $CODEDIR/work/<NEW_MODEL>/train-ssrn/
ln -s $CODEDIR/work/vctk_01/train-ssrn/model_epoch_4* ./work/<NEW_MODEL>/train-ssrn/
ln -s $CODEDIR/work/vctk_01/train-ssrn/checkpoint ./work/<NEW_MODEL>/train-ssrn/
```

## Synthesising

The only thing to note here is that the '-speaker' argument needs to be present, giving a speaker that was present during training:

```
./util/submit_tf.sh ./synthesize.py -c config/vctk_01.cfg -N 10 -speaker <SPEAKER>
```

## Description of existing configuration files:

- vctk_01: only adds speaker codes at the audio_decoder_input
- vctk_02: same ssrn as vctk_01, adds speaker codes at the audio_decoder_input AND text_encoder_towards_end
- vctk_03: learned channel contributions from the c.50 speakers

## Training a multispeaker model on single speaker data ("fine-tuning")

If wanting to fine-tune a multispeaker model to a single speaker (i.e. continue training), the most important differences are that in the new config file the 'speaker_list' must also include the new speaker. This also means that the 'nspeakers' needs to be one less than in the multispeaker model, e.g.:

```
nspeakers = len(speaker_list) + 99
```

Further, there needs to be an argument like the following, pointing to the trained multispeaker models:

```
initialise_weights_from_existing = [('Text2Mel', WORK+'/<OLD_MODEL>/train-t2m/model_epoch_<LAST_EPOCH>'), ('SSRN', WORK+'/<OLD_MODEL>/train-ssrn/model_epoch_<LAST_EPOCH>')]
```
