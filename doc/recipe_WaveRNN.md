
## DCTTS + WaveRNN

To generate DCTTS samples using WaveRNN set the following flag in your config file:
```
store_synth_features = True
```
and run the normal DCTTS synthesis script:
```
cd ophelia
dctts_synth_dir='/tmp/dctts_synth_dir/' 
./util/submit_tf.sh synthesize.py -c config/lj_tutorial.cfg -N 5 -odir ${dctts_synth_dir}
```

This saves the generated magnitude files (.npy) and Grifim-Lim wavefiles in the directory dctts_synth_dir.

To generate WaveRNN wavefiles from these magnitude files:
```
cd Tacotron
wavernn_synth_dir='/tmp/wavernn_synth_dir/'
python synthesize_dctts_wavernn.py -i ${dctts_synth_dir} -o ${wavernn_synth_dir}
```

## Notes on Tacotron+WaveRNN code installation

```
git clone https://github.com/cassiavb/Tacotron.git
cd Tacotron/
virtualenv --distribute --python=/usr/bin/python3.6 env
source env/bin/activate
pip install --upgrade pip
pip install torch torchvision
pip install -r requirements.txt 
pip install numba==0.48
```

