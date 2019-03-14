## Semisupervised training

### Babbler training

Train 'babbler' (300 epochs only):

```
./util/submit_tf.sh ./train.py -c ./config/lj_03.cfg -m babbler 
```

Note that this wasn't implemented when I trained the voice before - hope it works OK:

```
bucket_data_by = 'audio_length'  
```

In any case, text in transcript is ignored when training babbler.

Copy existing SSRN:

```
cp -rs /disk/scratch/oliver/dc_tts_osw_clean/work/lj_02/train-ssrn /disk/scratch/oliver/dc_tts_osw_clean/work/lj_03/
```

Synthesise by babbling:

```
./util/submit_tf.sh ./synthesize.py -c ./config/lj_03.cfg -babble
```

(Note: all sentences are currently seeded with the same start (all zeros from padding) so all babbled outputs will be identical)


### Fine tuning 

Fine tune with text as conventional model on transcribed subset (1000 sentences) of the data:

```
 ./util/submit_tf.sh ./train.py -c ./config/lj_05.cfg -m t2m  

cp -rs /disk/scratch/oliver/dc_tts_osw_clean/work/lj_02/train-ssrn /disk/scratch/oliver/dc_tts_osw_clean/work/lj_05/

 ./util/submit_tf.sh ./synthesize.py -c ./config/lj_05.cfg -N 10
```

### Baseline

Compare training from scratch on 1000 sentences:

```
 ./util/submit_tf.sh ./train.py -c ./config/lj_04.cfg -m t2m     

cp -rs /disk/scratch/oliver/dc_tts_osw_clean/work/lj_02/train-ssrn /disk/scratch/oliver/dc_tts_osw_clean/work/lj_04/

 ./util/submit_tf.sh ./synthesize.py -c ./config/lj_04.cfg -N 10

```



