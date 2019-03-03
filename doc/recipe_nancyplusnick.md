
# Train on Nancy + Nick



## Combine Nancy & Nick data already used for nancy_01 and nancy2nick_*


Combine transcripts, adding speaker codes:
```
DATADIR=/group/project/cstr2/owatts/data/nick_plus_nancy
mkdir $DATADIR

grep -v ^$ /group/project/cstr2/owatts/data/nick16k/transcript.csv | awk '{print $0"|nick"}' > $DATADIR/transcript.csv
grep -v ^$ /group/project/cstr2/owatts/data/nancy/derived/transcript.csv | awk '{print $0"|nancy"}' | grep -v NYT00 >> $DATADIR/transcript.csv

# (remove empty lines and NYT00 section for which attention guides were not made)

cp /group/project/cstr2/owatts/data/nick16k/transcript_{mrt,hvd}.csv $DATADIR
```

Combine acoustic features and attention guides:

```
mkdir -p work/nancyplusnick_01/data/{attention_guides,full_mels,mels,mags}/
for SUBDIR in attention_guides full_mels mels mags ; do
  for VOICE in nancy2nick_01 nancy_01 ; do
    ln -s ${PWD}/work/$VOICE/data/$SUBDIR/* work/nancyplusnick_01/data/$SUBDIR/ ;
  done
done
```



Prepare config file and train:

```
./util/submit_tf.sh ./train.py -c ./config/nancyplusnick_01.cfg -m t2m
```

Previously-trained SSRN can be softlinked without retraining: 
```
cp -rs  `pwd`/work/nancy2nick_01/train-ssrn/ `pwd`/work/nancyplusnick_01/train-ssrn/
```


Synth

```
./util/submit_tf.sh ./synthesize.py -c ./config/nancyplusnick_01.cfg -N 10 -speaker nick
./util/submit_tf.sh ./synthesize.py -c ./config/nancyplusnick_01.cfg -N 10 -speaker nancy
```




Fine tune on nick only:

./util/submit_tf.sh ./train.py -c ./config/nancyplusnick_02.cfg -m t2m
cp -rs  $PWD/work/nancy2nick_01/train-ssrn/ ./work/nancyplusnick_02/train-ssrn/



 ./util/submit_tf.sh ./synthesize.py -c ./config/nancyplusnick_02.cfg     -N 10 -speaker nick



 set attention loss weight very high:


 ```
 ./util/submit_tf.sh ./train.py -c ./config/nancyplusnick_03.cfg -m t2m
cp -rs  $PWD/work/nancy2nick_01/train-ssrn/ ./work/nancyplusnick_03/train-ssrn/



 ./util/submit_tf.sh ./synthesize.py -c ./config/nancyplusnick_03.cfg     -N 10 -speaker nick


 ```



 Try learning channel contributions for each speaker:

 ./util/submit_tf.sh ./train.py -c ./config/nancyplusnick_04_lcc.cfg -m t2m ; cp -rs  $PWD/work/nancy2nick_01/train-ssrn/ ./work/nancyplusnick_04_lcc/train-ssrn/





 Try learning channel contributions for each speaker + SD-phone embeddings:

 ./util/submit_tf.sh ./train.py -c ./config/nancyplusnick_05_lcc_sdpe.cfg -m t2m ; cp -rs  $PWD/work/nancy2nick_01/train-ssrn/ ./work/nancyplusnick_04_lcc/train-ssrn/


