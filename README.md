# dc_tts_osw

A modified version of Kyubyong Park's [dc_tts repository](https://github.com/Kyubyong/dc_tts), which implements a variant of the system described in [Efficiently Trainable Text-to-Speech System Based on Deep Convolutional Networks with Guided Attention](https://arxiv.org/abs/1710.08969). 

TODO: note differences between the paper and Kyubyong Park's implementation

TODO: Note differences between this and Kyubyong Park's version




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





## Prepare data

As an example, download the LJSpeech database:

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







### Extract acoustic features 


<!-- depnds on variables:

    coarse_audio_dir = topworkdir + '/' + config_name + '/data/mels/'  
    full_audio_dir = topworkdir + '/' + config_name + '/data/mags/' 
    extract_full_mel = topworkdir + '/' + config_name + '/data/full_mels/' 

 -->
```
./util/submit_tf_cpu.sh ./prepo.py -c ./config/lj_01.cfg -ncores 25
```

Will output data by default in directories under:

```
$CODEDIR/work/<config>/data/
```

This can take quite a bit of space:

```
du -sh work/lj_01/data/*
2.1G    work/lj_01/data/full_mels
27G     work/lj_01/data/mags
548M    work/lj_01/data/mels
```

... so alternatively just softlink from older features extracted with compatible config:

```
cp -rs $CODEDIR/work/<old_config_name>/data/ $CODEDIR/work/<new_config_name>/
```


NB:  (random shift in reduction):







### Configure lengths of data

Now we have extracted features, inspect dataset again:

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






### Make per-utterance attention guides

Configuring:

```
attention_guide_dir = ''
```

... will use a single global attention guide as in Park's original code, of size `(max_N, max_T)`. This might be bad if there is much difference in rate between sentences, as would be the case in multispeaker databases. Use this command to prepare per-utterance guides: 

```
./util/submit_tf.sh ./prepare_attention_guides.py -c ./config/lj_01.cfg -ncores 20
```

Config auto?



### Hold out some validation material (NB: SD / SI):
To hold out 50th chapter of LJ:

```
validpatt = 'LJ050-'
```

## Train
./util/submit_tf.sh ./train.py -c ./config/lj_01.cfg -m 1
./util/submit_tf.sh ./train.py -c ./config/lj_01.cfg -m 2












## Synthesise:


Make test transcript:

...


Copy preexisting test transcript:

```
cp /disk/scratch/oliver//data/nick_hurricaine/test_transcript.csv  /disk/scratch/oliver//data/LJSpeech-1.1/
```



## default: uses the last saved models (best according to validation):-
./util/submit_tf.sh ./synthesize.py -c ./config/lj_01.cfg -N 10



## synthesise from validation parameters stored at all epochs of training:

./util/submit_tf.sh synthesise_validation_waveforms.py -c ./config/lj_01.cfg 






===

cd /disk/scratch/oliver/data/nick
cp -rs  /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/wav  ./wav/


rm -rf /disk/scratch/oliver/data/nick/transcript_1.csv
for TXT in /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/txt/herald* ; do
    BASE=`basename $TXT .txt`;
    TEXT=`cat $TXT` ;
    echo $BASE\|\|$TEXT >> /disk/scratch/oliver/data/nick/transcript_1.csv
done 


 ./util/submit_tf_cpu.sh ./prepo.py -c ./config/nick_01.cfg -ncores 25



 ------
 VCTK data:

 (waffler) [stardale]owatts: ./util/submit_tf_cpu.sh ./script/split_speech.py -w $DATADIR/wav_norm/ -o $DATADIR/wav_trim/ -dB 15 -ncores 25 -trimonly
===









## Sync to server

It will probably be most convenient to have a central copy of the repository in AFS space, and to synchronise this with local copies on the GPU machines you are using.  The script `sync_code_from_afs.sh` can be used to keep various versions in synch without pushing to git. E.g., if you are wor



```
mkdir /disk/scratch/oliver/dc_tts_osw
  510  cd /disk/scratch/oliver/dc_tts_osw
  511  mkdir util
  512  cp ~/repos/dc_tts_osw/util/sync_code_from_afs.sh ./util/
  513  ./util/sync_code_from_afs.sh

```

## 





get data


mkdir /disk/scratch/oliver/data/roger/
cp -rs  /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/CSTR-Private/Roger/peter-release/48kHz/ /disk/scratch/oliver/data/roger/wav/


python ./script/normalise_level.py -i /disk/scratch/oliver/data/roger/wav/ -o /disk/scratch/oliver/data/roger/wav_norm/ -ncores 30


./util/submit_tf_cpu.sh ./script/split_speech.py -w /disk/scratch/oliver/data/roger/wav_norm/ -o /disk/scratch/oliver/data/roger/wav_trim/ -dB 30 -ncores 30 -trimonly

### get transcript


cp /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/roger/transcript.csv /disk/scratch/oliver/data/roger/transcript.csv 

## get test trascnipt:

 cp /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/test_transcript.csv /disk/scratch/oliver/data/roger/

### check transcript:

python ./script/check_transcript.py -i /disk/scratch/oliver/data/roger/transcript.csv -phone

### Based on printed 'Observed phones' edit config's vocab (NB: insert <PAD> in 0th position!)

### get mel

./util/submit_tf_cpu.sh ./prepo.py -c ./config/roger_01.cfg -ncores 20


### remove some subsets of sentences:



### check transcript again now we have audio to look at sentence lengths:

python ./script/check_transcript.py -i /disk/scratch/oliver/data/roger/transcript.csv -phone -cmp ./work/roger_mags


## train 

./util/submit_tf.sh ./train.py -c ./config/roger_01.cfg -m 1




## synthesise dev set


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/roger_02.cfg -m 2



## synth mels of all archived T2M epochs:--
(waffler) [stardale]owatts: ./util/submit_tf.sh synth_sweep_devset.py -c ./config/roger_02.cfg -m 1

## synth waves with all SSRN epochs, from oracle mels:--
 /disk/scratch/oliver/dc_tts_osw/work//roger_02/train-2/archive/model_gs_380k


##  
./util/submit_tf.sh synth_sweep_devset.py -c ./config/roger_02.cfg -m 1 -ssrn  /disk/scratch/oliver/dc_tts_osw/work//roger_02/train-2/archive/model_gs_380k





## ==== PRETRAIN  MULTISPAKER MODEL ON VCTK ======

Data here:  /group/project/cstr2/owatts/vctk_voices_ossian/VCTK-Corpus/

Make VCTK transcripts:-
```
mkdir /disk/scratch/oliver/data/vctk

python ./script/datasets/gather_vctk_transcript.py -i /group/project/cstr2/owatts/vctk_voices_ossian/VCTK-Corpus/txt/ -o /disk/scratch/oliver/data/vctk/utts.data
```


cd /disk/scratch/oliver/data/vctk/

FEST=/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/tool/festival/festival/bin/festival
SCRIPT=/afs/inf.ed.ac.uk/user/o/owatts/repos/dc_tts_osw/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript1.csv

cd /disk/scratch/oliver/dc_tts_osw/

 python script/check_transcript.py -i /disk/scratch/oliver/data/vctk/transcript1.csv  -phone

python script/festival/fix_transcript.py  /disk/scratch/oliver/data/vctk/transcript1.csv  B C I It K M N P R She V age dot imposition position there wide   > /disk/scratch/oliver/data/vctk/transcript.csv 


## add spkr:

[stardale]owatts: awk -F_ '{print $1}' /disk/scratch/oliver/data/vctk/transcript.csv > /tmp/spkr.txt
[stardale]owatts: mv  /disk/scratch/oliver/data/vctk/transcript.csv  /disk/scratch/oliver/data/vctk/transcript.csv
transcript1.csv  transcript.csv
[stardale]owatts: mv  /disk/scratch/oliver/data/vctk/transcript.csv  /disk/scratch/oliver/data/vctk/transcript2.csv
[stardale]owatts: paste -d |  /disk/scratch/oliver/data/vctk/transcript2.csv /tmp/spkr.txt > /disk/scratch/oliver/data/vctk/transcript.csv
-bash: /disk/scratch/oliver/data/vctk/transcript2.csv: Permission denied
paste: option requires an argument -- 'd'
Try 'paste --help' for more information.
[stardale]owatts: paste -d'|'  /disk/scratch/oliver/data/vctk/transcript2.csv /tmp/spkr.txt > /disk/scratch/oliver/data/vctk/transcript.csv





### audio
# links already here:

# /group/project/cstr2/owatts/vctk_voices_ossian/wav/


python ./script/normalise_level.py -i /group/project/cstr2/owatts/vctk_voices_ossian/wav/ -o /disk/scratch/oliver/data/vctk/wav_norm/ -ncores 25

./util/submit_tf_cpu.sh ./script/split_speech.py -w /disk/scratch/oliver/data/vctk/wav_norm/ -o /disk/scratch/oliver/data/vctk/wav_trim/ -dB 30 -ncores 25 -trimonly

### get mel (after transcript in place)

./util/submit_tf_cpu.sh ./prepo.py -c ./config/vctk_01.cfg -ncores 25

### check legnths an phoneset
python script/check_transcript.py -i /disk/scratch/oliver/data/vctk/transcript.csv  -phone -cmp ./work/vctk_mags/

python script/check_transcript.py -i /disk/scratch/oliver/data/vctk/transcript.csv  -phone -cmp ./work/vctk_mags/ -maxframes 400 -maxletters 80 -speaker



HERE!!!!!!!!!!!!



 ./util/submit_tf.sh  ./train.py -c ./config/vctk_01.cfg -m 1
  ./util/submit_tf.sh  ./train.py -c ./config/vctk_01.cfg -m 2



  : cp  /disk/scratch/oliver/data/roger/test_transcript.csv   /disk/scratch/oliver/data/vctk 

  ./util/submit_tf.sh  ./synthesize.py -c ./config/vctk_01.cfg -speaker p225





===== nancy phones =====

(waffler) [starleader]owatts: pwd
/disk/scratch/oliver/data/nancy


cp prompts.data utts.data




### needed to modify phoneset to add o~
emacs  /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/tool/festival/festival/lib/combilex_phones.scm


FEST=/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/tool/festival/festival/bin/festival
SCRIPT=/afs/inf.ed.ac.uk/user/o/owatts/repos/dc_tts_osw/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript1.csv



cd /disk/scratch/oliver/dc_tts_osw/

 python script/check_transcript.py -i /disk/scratch/oliver/data/nancy/transcript1.csv  -phone

python script/festival/fix_transcript.py  /disk/scratch/oliver/data/nancy/transcript1.csv A African B Bradley COLUMN Constitution D Dyson E Gore Herald I Israel Joyner Kettering L Ledger Lewinsky M Marting Marts Mifflin Minimalism Moraine N O Packard ROM S Saturdays Speech T Time To Torricelli Two U V Walker War Weaver West Wolf World a access actualized adolescents adult adulthood after age alone along always american americanism and arranger atlantic author authored awaited away axis bang banging barr based bats be bearing bed bedroom being benefit berries birds black blended blood bodied born borne bottomed bound boxing boy brained breaking breathers bred broadway brow building built burner busting calorie cancer candid card care carpenter carpet case catch cell cells center centeredness central cents century chalk changing charge chief child chin class clearing clock closing club cock colored colour company comte concrete cone conserving consistently constitution constructing control counseling counter couture d dance daughter day death deceiving decorated deeper degradation demand democratic destruction detecting diamonds diet digit dimensional disabled diversity do doctoral dog doing dollar dollars door dormant dramatic drenched dried driven drug dumping duty dwelling ear eating ed edge edged effects efficient eight elite emphasize encrusted engage englishman enter enterprises epilepsy eroding ever evolved evolving examination executivedom expressing eye eyed fabric face facing falling farm fascist fat fault feast feathered fed feeder feeling fetus fever fictional field fifth file filled finance fire fired first fishing five-hour flying focused food foot footed for forbidden foreman founder four fourth free friendly front frontal fruit funded game gas generated generation generational ghali girlfriend go gold government grader grandchildren grandparents grass green ground growing guard guidance h haired hand handed hating head headed health heavy heels hero high hill hit home hop hoped hops hormonal host hours human hundred in inch income indicative induced indulgent industries industry infant infective intelligence investment iron it january jewish job jobs july justice key kinesthetic knacks knife laden largest lashed lasting law leading led leg legged less lettered level life like liked limbed limited line liners list lit lived long looking loving low m making man management managers manufactured marketed married mathematical may mellon menopausal miller million minded minute miracles mob moded modified month morning motivation mouthed mover naked native neighbors nested neutral news nineteenth nonsense nosed nursing obsessed of off offered offs old on only opener opening ops optic or ordered organization organized ourselves out outside over owned paid painted pan par part party paying pearl peer per pitched placement planned plate policy polluted poor pots poverty president pressure price print producing production profile profit profits prone protector publicized racer radio rail raiser raisers ramps ranch range ranking rare rate ray raying read real received red referred reflective rejection rejuvenating related reliant replacement report reproducing resident restricted review reviewed rex righteous road rock rocketing rotating rubbing run running salted satisfied saturated savoy scan school scoring sealed seated secondary seizure september sergeant service sessions setting settled seventies sex shaped share sharing sharp shattering shaven shell shelled shot shouldered show sided sit sitting size sized smart smelling sniffing society soil soldier sour specific speech spend spirit spoken spouse spouses square stage staid star started stealing steer step sticky still stitched stocked stomach stop striped strong studded studies style suk sulphur supervisory support swept tag tailed talk tea tech temps ten tension tent than the therapy thick thieves thin thinking thousand threatening thrill through throw thrust thumping tiled timbered time tipped to tolerant tome top tough town trade training transmitting tricarico tries tse tuned twentieth two typical unit up upmanship ups using veto view voiced wage watch wave way wear web weird western whaling white wide wife willed win wink wise with won worker workers worthy ya year years your zionist zip zone  > /disk/scratch/oliver/data/nancy/transcript_combilex2.csv 

 cp -rs /disk/scratch/oliver/dc_tts/work/nancy_mels ./work/nancy_GL_lett

(waffler) [starleader]owatts: cp -rs /disk/scratch/oliver/dc_tts/work/nancy_mags ./work/nancy_mags


 python script/check_transcript.py -i /disk/scratch/oliver/data/nancy/transcript_combilex2.csv -phone -cmp ./work/nancy_mels 

### remove utts with no audio:
 python script/check_transcript.py -i /disk/scratch/oliver/data/nancy/transcript_combilex2.csv -phone -cmp ./work/nancy_mels -o /disk/scratch/oliver/data/nancy/transcript_combilex.csv


## train
....



### synth various combinations


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nancy_phones_01.cfg -t2m ./work/nancy_phones_01/train-1/archive/model_gs_340k -ssrn ./work/nancy_phones_01/train-2/archive/model_gs_110k -mode synthesis -n 5


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nancy_phones_01.cfg -t2m ./work/nancy_phones_01/train-1/archive/model_gs_340k -ssrn ./work/nancy_phones_01/train-2/archive/model_gs_110k -mode synthesis -n 5



### fine tune on nick starting at nancy T2M : model_gs_050k (early model) and SSRN model_gs_110k (later model)

 ./util/submit_tf.sh ./train.py -c ./config/nick_phones_01.cfg -m 1


 ./util/submit_tf.sh ./train.py -c ./config/nick_phones_01.cfg -m 1
  ./util/submit_tf.sh ./train.py -c ./config/nick_phones_01.cfg -m 2

## synth nick

 ./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_01.cfg -t2m ./work/nick_phones_01/train-1/archive/model_gs_018k -ssrn ./work/nick_phones_01/train-2/archive/model_gs_007k -mode synthesis -n 5



## error! not restarting from nancy. Try again:


 ./util/submit_tf.sh ./train.py -c ./config/nick_phones_02.cfg -m 1
  ./util/submit_tf.sh ./train.py -c ./config/nick_phones_02.cfg -m 2



./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_02.cfg -t2m ./work/nick_phones_02/train-1/archive/model_gs_001k -ssrn ./work/nick_phones_02/train-2/archive/model_gs_111k -mode synthesis -n 5


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_02.cfg -t2m ./work/nick_phones_02/train-1/archive/model_gs_030k -ssrn ./work/nick_phones_02/train-2/archive/model_gs_112k -mode synthesis -n 5

./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_02.cfg -t2m ./work/nick_phones_02/train-1/archive/model_gs_050k -ssrn ./work/nick_phones_02/train-2/archive/model_gs_115k -mode synthesis -n 5


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_02.cfg -t2m ./work/nick_phones_02/train-1/archive/model_gs_040k -ssrn ./work/nick_phones_02/train-2/archive/model_gs_115k -mode synthesis -n 5





### final hvd (alter test script in config):


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_02.cfg -t2m ./work/nick_phones_02/train-1/archive/model_gs_150k -ssrn ./work/nick_phones_02/train-2/archive/model_gs_138k -mode synthesis 

### final mrt (alter test script in config):


./util/submit_tf.sh ./synth_sweep_devset.py -c ./config/nick_phones_02.cfg -t2m ./work/nick_phones_02/train-1/archive/model_gs_150k -ssrn ./work/nick_phones_02/train-2/archive/model_gs_138k -mode synthesis 


## GL:

 ./util/submit_tf_cpu.sh ./copy_synth_GL.py -c ./config/nick_phones_testset_features.cfg -o ./work/nick_resynth_GL_hvd_16bit

## ----- nick combilex trainscript



cd /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/


mv transcript.csv transcript.csv.unilex
mv transcript2.csv  transcript2.csv.unilex



FEST=/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/tool/festival/festival/bin/festival
SCRIPT=/afs/inf.ed.ac.uk/user/o/owatts/repos/dc_tts_osw/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript1.csv

cd /disk/scratch/oliver/dc_tts_osw/


 python script/check_transcript.py -i /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/transcript1.csv  -phone

python script/festival/fix_transcript.py  /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/transcript1.csv B C K P V imposition wide > /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/transcript_combilex2.csv 



 611  mv test_transcript.csv test_transcript.csv.unilex
  612  mv test_transcript1.csv test_transcript1.csv.unilex
  613  FEST=/afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/tool/festival/festival/bin/festival
  614  SCRIPT=/afs/inf.ed.ac.uk/user/o/owatts/repos/dc_tts_osw/script/festival/make_rich_phones_combirpx_noplex.scm
  615  $FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript1.csv
  616   python script/check_transcript.py -i /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/transcript1.csv  -phone
  617  cd /disk/scratch/oliver/dc_tts_osw/
  618   python script/check_transcript.py -i /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/transcript1.csv  -phone
  619  python script/festival/fix_transcript.py  /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/transcript1.csv cross > /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/transcript_combilex2.csv
  620   python script/check_transcript.py -i /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/transcript_combilex2.csv -phone




## get test audio:

wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/347/quiet_mrt.zip?sequence=3&isAllowed=y
wget https://datashare.is.ed.ac.uk/bitstream/handle/10283/347/quiet_harvard.zip?sequence=17&isAllowed=y

unzip quiet_mrt.zip\?sequence\=3
unzip quiet_harvard.zip\?sequence\=17
mkdir wav
mv mrt_*wav wav/
mv hvd_*wav wav/

## rename test audio, remove take (?) numbers:
```
mkdir /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/wav_renamed
ls /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/wav/*wav | while read WAV ; do 
    BASE=`basename $WAV .wav`;
    echo $BASE ;
    PART1=`echo $BASE | awk -F_ '{print $1}'`
    PART2=`echo $BASE | awk -F_ '{print $2}'`
    
    BASE3="${PART1}_${PART2}"
    echo $BASE3
    ln -s $WAV /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/wav_renamed/$BASE3.wav
done
```



### waves and transcript not named consistent -- fix this:
```
mkdir /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/wav_renamed
ls /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/wav/*wav | while read WAV ; do 
    BASE=`basename $WAV .wav`;
    echo $BASE ;
    BASE2=`echo $BASE | awk -F_ '{print $1}'`
    echo $BASE2
    BASE3=$(echo $BASE2 | sed 's/^0*//')
    echo $BASE3
    ln -s $WAV /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/wav_renamed/$BASE3.wav
done
```


cp /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/transcript_combilex2.csv  /disk/scratch/oliver/data/nick_hurricaine/transcript.csv 

cp /afs/inf.ed.ac.uk/user/o/owatts/sim2/oliver/data/dctts/nick_hurricaine/test/transcript_combilex2.csv  /disk/scratch/oliver/data/nick_hurricaine/test_transcript.csv 

 ./util/submit_tf_cpu.sh ./prepo.py -c ./config/nick_phones_01.cfg -ncores 25




