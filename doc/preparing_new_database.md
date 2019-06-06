

## Preparing a new database

This describes how the features at `/group/project/cstr2/owatts/data/LJSpeech-1.1/` were made and will be useful for preparing new databases. To prepare the phonetic transcription from text, Festival's front-end was used: see notes on [installing Festival](./festival_install.md).


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

