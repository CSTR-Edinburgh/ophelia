# Voices for SSW10 experiment

Instructions for recreating SSW10 voices


## Tools


### DCTTS code
```
TOOLDIR=/disk/scratch/script_project/ssw10/tools/
mkdir -p $TOOLDIR
cd $TOOLDIR
git clone https://github.com/oliverwatts/dc_tts_osw.git dc_tts_osw_A
cd dc_tts_osw_A
```

### Festival 

```
INSTALL_DIR=$TOOLDIR/festival
mkdir -p $INSTALL_DIR
cd $INSTALL_DIR

wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festival-2.4-release.tar.gz
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/speech_tools-2.4-release.tar.gz

tar xvf festival-2.4-release.tar.gz
tar xvf speech_tools-2.4-release.tar.gz

cd speech_tools
./configure  --prefix=$INSTALL_DIR
gmake

cd ../festival
./configure  --prefix=$INSTALL_DIR
gmake

cd $INSTALL_DIR

wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/voices/festvox_cmu_us_awb_cg.tar.gz
tar xvf festvox_cmu_us_awb_cg.tar.gz

## Get lexicons for the english voice:
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_CMU.tar.gz
tar xvf festlex_CMU.tar.gz
 
wget http://www.cstr.ed.ac.uk/downloads/festival/2.4/festlex_POSLEX.tar.gz
tar xvf festlex_POSLEX.tar.gz

##  test
cd $INSTALL_DIR/festival/bin

## run the *locally installed* festival (NB: initial ./ is important!)
./festival

festival> (voice_cmu_us_awb_cg)
festival> (utt.save.wave (SayText "If i'm speaking then installation actually went ok.") "test.wav" ' riff)
```







## Data


### Download LJSpeech

```
DATADIR=/disk/scratch/script_project/ssw10/data
mkdir -p $DATADIR

cd $DATADIR
wget http://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
bunzip2  LJSpeech-1.1.tar.bz2
tar xvf LJSpeech-1.1.tar
rm LJSpeech-1.1.tar
```


### Phonetise the transcription with Festvial + CMU lexicon

```
CODEDIR=/disk/scratch/oliver/dc_tts_osw/
cd $CODEDIR
python ./script/festival/csv2scm.py -i $DATADIR/LJSpeech-1.1/metadata.csv -o $DATADIR/LJSpeech-1.1/utts.data

cd $DATADIR/LJSpeech-1.1/
FEST=$TOOLDIR/festival/festival/bin/festival
SCRIPT=$CODEDIR/script/festival/make_rich_phones_cmulex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp2.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp2.csv > ./transcript_cmu.csv
```

