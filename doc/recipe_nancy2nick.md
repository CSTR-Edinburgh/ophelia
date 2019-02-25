
# Very naive speaker adaptation to convert Nancy to Nick

The simplest way to train on a small database is to fine tune
a speaker-dependent voice to the new database. This works
surprisingly well even where the base voice is of a different 
sex and accent to the target speaker, as this example shows.


## Prepare nick data



We will use a version of the nick data which has been downsampled to 
16kHz with sox:

```
/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/ICPhS19/samples/second_submission/natural_16k/ 
```

It was converted from the 48kHz version here:

```
/afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/wav/ 
```

### waveforms
```
INDIR=/afs/inf.ed.ac.uk/group/cstr/projects/nst/cvbotinh/SCRIPT/ICPhS19/samples/second_submission/natural_16k/
OUTDIR=/group/project/cstr2/owatts/data/nick16k/

python ./script/normalise_level.py -i $INDIR -o $OUTDIR/wav_norm/ -ncores 25

./util/submit_tf_cpu.sh ./script/split_speech.py -w $OUTDIR/wav_norm/ -o $OUTDIR/wav_trim/ -dB 30 -ncores 25 -trimonly
```

### transcript

Gather texts:

```
for FNAME in /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/txt/herald_* ; do 
  BASE=`basename $FNAME .txt` ; 
  TEXT=`cat $FNAME` ; 
  echo "${BASE}||${TEXT}" ; 
done > /group/project/cstr2/owatts/data/nick16k/metadata.csv


for FNAME in /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/txt/hvd_* ; do 
  BASE=`basename $FNAME .txt` ; 
  TEXT=`cat $FNAME` ; 
  echo "${BASE}||${TEXT}" ; 
done > /group/project/cstr2/owatts/data/nick16k/metadata_hvd.csv


for FNAME in /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/txt/mrt_* ; do 
  BASE=`basename $FNAME .txt` ; 
  TEXT=`cat $FNAME` ; 
  echo "${BASE}||${TEXT}" ; 
done > /group/project/cstr2/owatts/data/nick16k/metadata_mrt.csv
```

Phonetise:

```
CODEDIR=`pwd`
DATADIR=/group/project/cstr2/owatts/data/nick16k/
python ./script/festival/csv2scm.py -i $DATADIR/metadata.csv -o $DATADIR/utts.data

cd $DATADIR/
FEST=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/tool/festival/festival/src/main/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv
python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript.csv


cd $CODEDIR
for TESTSET in mrt hvd ; do
    mkdir /group/project/cstr2/owatts/data/nick16k/test_${TESTSET}
    python ./script/festival/csv2scm.py -i $DATADIR/metadata_${TESTSET}.csv -o $DATADIR/test_${TESTSET}/utts.data
done


for TESTSET in mrt hvd ; do
    cd /group/project/cstr2/owatts/data/nick16k/test_${TESTSET}
    $FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv
    python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript.csv
    cp transcript.csv ../transcript_${TESTSET}.csv
done

```




### features
```
./util/submit_tf_cpu.sh ./prepare_acoustic_features.py -c ./config/nancy2nick_01.cfg -ncores 15
./util/submit_tf.sh ./prepare_attention_guides.py -c ./config/nancy2nick_01.cfg -ncores 15
```


## training

Config `nancy2nick_01` updates all weights pretrained on the Nancy data:

```
./util/submit_tf.sh ./train.py -c ./config/nancy2nick_01.cfg -m t2m
./util/submit_tf.sh ./train.py -c ./config/nancy2nick_01.cfg -m ssrn
```

Config `nancy2nick_01` updates all weights pretrained on the Nancy data, except
those of textencoder which are kept frozen:

```
./util/submit_tf.sh ./train.py -c ./config/nancy2nick_02.cfg -m t2m
```

Previously-trained SSRN can be softlinked without retraining: 
```
cp -rs  `pwd`/work/nancy2nick_01/train-ssrn/ `pwd`/work/nancy2nick_02/train-ssrn/
```

## Synthesis
```
./util/submit_tf.sh ./synthesize.py -c config/nancy2nick_01.cfg 
./util/submit_tf.sh ./synthesize.py -c config/nancy2nick_02.cfg
```

