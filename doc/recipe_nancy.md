
# Blizzard Nancy corpus preparation and voice building

For base voice for ICPhS in first instance.


```
### get the (publicly downloadable) data from CSTR datawww:-
mkdir /group/project/cstr2/owatts/data/nancy/
cd /group/project/cstr2/owatts/data/nancy/
mkdir original

cp /afs/inf.ed.ac.uk/group/cstr/datawww/blizzard2011/lessac/wavn.tgz ./original/
cp /afs/inf.ed.ac.uk/group/cstr/datawww/blizzard2011/lessac/lab.ssil.zip ./original/
cp /afs/inf.ed.ac.uk/group/cstr/datawww/blizzard2011/lessac/lab.zip ./original/
cp /afs/inf.ed.ac.uk/group/cstr/datawww/blizzard2011/lessac/prompts.data ./original/

### compare checksums with published ones:
md5sum ./original/*
0a4860a69bca56d7e9f8170306ff3709  ./original/lab.ssil.zip
aeae7916d881a8eef255a6fe05e77e77  ./original/lab.zip
650b44f7252aed564d190b76a98cb490  ./original/prompts.data
bb2a80dd1423f87ba12d2074af8e7a3f  ./original/wavn.tgz

### ...all OK.

cd ./original
tar xvf wavn.tgz
unzip lab.ssil.zip
unzip lab.zip

rm *.zip
rm *.tgz

### information about the data:-
http://www.cstr.ed.ac.uk/projects/blizzard/2011/lessac_blizzard2011/:

* prompts.data - File with all of the prompt texts in filename order.
* corrected.gui - File with all of the prompts in Lesseme labeled format in the order of the Nancy corpus as originally recorded, after some labels produced by the Lessac Front-end were corrected to reflect what the voice actor actually said.
* uncorrected.gui - File with all of the prompts in Lesseme labeled format in the order of the Nancy corpus as produced by the Lessac Front-end from the prompts.data file without correction to the labels for what the voice actor actually said.
* lab.ssil.zip - Zipped set of files with Lesseme labels that include the result of automated segmentation of the Lesseme labels in the corrected.gui file before the ssil label is collapsed into the preceding or following label.
* lab.zip 

cd /disk/scratch/oliver/dc_tts_osw_clean
mkdir /group/project/cstr2/owatts/data/nancy/derived/
python ./script/normalise_level.py -i /group/project/cstr2/owatts/data/nancy/original/wavn/ \
     -o /group/project/cstr2/owatts/data/nancy/derived/wav_norm/ -ncores 25

./util/submit_tf_cpu.sh ./script/split_speech.py \
        -w /group/project/cstr2/owatts/data/nancy/derived/wav_norm/ \
        -o /group/project/cstr2/owatts/data/nancy/derived/wav_trim/ -dB 30 -ncores 25 -trimonly

rm -r /group/project/cstr2/owatts/data/nancy/derived/wav_norm/        



## transcription (needed to add o~ to combilex rpx phoneset in Festival):-

## use existing scheme format transcript:-
cp /group/project/cstr2/owatts/data/nancy/original/prompts.data /group/project/cstr2/owatts/data/nancy/derived/utts.data
cd /group/project/cstr2/owatts/data/nancy/derived/

CODEDIR=/disk/scratch/oliver/dc_tts_osw_clean
FEST=/afs/inf.ed.ac.uk/group/cstr/projects/simple4all_2/oliver/tool/festival/festival/src/main/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_combirpx_noplex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./transcript.csv



### get phone list to ad to config:
 python ./script/check_transcript.py -i /group//project/cstr2/owatts/data/nancy/derived/transcript.csv -phone



./util/submit_tf_cpu.sh ./prepare_acoustic_features.py -c ./config/nancy_01.cfg -ncores 25

./util/submit_tf.sh ./prepare_attention_guides.py -c ./config/nancy_01.cfg -ncores 25


## train
./util/submit_tf.sh ./train.py -c config/nancy_01.cfg -m t2m
./util/submit_tf.sh ./train.py -c config/nancy_01.cfg -m ssrn
```