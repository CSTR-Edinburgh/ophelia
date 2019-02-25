

## Multispeaker system -- WORK IN PROGRESS, NOT YET FIT FOR CONSUMPTION

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



## make transcript of harvard sentences for testing:
mkdir /group/project/cstr2/owatts/data/VCTK-Corpus/test_set
cd /group/project/cstr2/owatts/data/VCTK-Corpus/test_set
DATADIR=`pwd`
CODEDIR=/disk/scratch/oliver/dc_tts_osw_clean

rm -f harvard.csv
ls /afs/inf.ed.ac.uk/group/cstr/projects/corpus_1/Nick48kHz/txt/hvd*.txt | while read TXT ; do 
      BASE=`basename $TXT .txt`
      LINE=`cat $TXT` ; 
      echo "$BASE||$LINE"; 
done >> harvard.csv

cd $CODEDIR
python ./script/festival/csv2scm.py -i $DATADIR/harvard.csv -o $DATADIR/utts.data

cd $DATADIR/
FEST=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/tool/festival/festival/src/main/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./harvard_tmp.csv

python $CODEDIR/script/festival/fix_transcript.py ./harvard_tmp.csv > ./harvard_combilex_rpx.csv

cp harvard_combilex_rpx.csv ..
cp harvard.csv ..


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

./util/submit_tf.sh ./train.py -c config/vctk_01.cfg -m t2m
./util/submit_tf.sh ./train.py -c config/vctk_01.cfg -m ssrn




######## speaker codes at 2 locations (data & ssrn same as above):


./util/submit_tf.sh ./train.py -c ./config/vctk_02.cfg -m t2m


## share ssrn by softlinking:
mkdir  work/vctk_02/train-ssrn/
ln -s  `pwd`/work/vctk_01/train-ssrn/model_epoch_4* ./work/vctk_02/train-ssrn/

ln -s  `pwd`/work/vctk_01/train-ssrn/checkpoint ./work/vctk_02/train-ssrn/



```

