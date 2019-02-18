

# Installing Festival

## Basic Festival install of Spanish and Scottish voices
```
INSTALL_DIR=/afs/some/convenient/directory/festival 

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

## Get spanish voice:
wget http://festvox.org/packed/festival/1.4.1/festvox_ellpc11k.tar.gz
tar xvf festvox_ellpc11k.tar.gz

## Get scottish english voice:
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
festival> (voice_el_diphone )
festival> (SayText "La rica salsa canaria se llama mojo pic'on.")

festival> (voice_cmu_us_awb_cg)
festival> (SayText "If i'm speaking then installation actually went ok.")


## synthesise to file:

(utt.save.wave (SynthText "La rica salsa canaria se llama mojo pic'on.")  "/path/to/output/file.wav" 'riff)
```


## Combilex installation

Given the file `combilex.tar` which contains 3 combilex surface form lexicons (gam, rpx, edi), install like this:

``` 
cp combilex.tar $INSTALL_DIR/festival/
cd $INSTALL_DIR/festival/
tar xvf combilex.tar
```

For processing the Nancy data, which contains a French word with a nasalised vowel present in the lexicon but not the phoneset definition, I needed to edit `$INSTALL_DIR/festival/lib/combilex_phones.scm` and add the line:

```
   (o~     + l 2 3 + n 0 0)   ;; added missing nasalised vowel
```

after the line:

```
   (@U     + d 2 3 + 0 0 0) ;ou
```


## Cleaning up

```
cd $INSTALL_DIR
rm *.tar.gz

cd $INSTALL_DIR/festival
rm *.tar
```


## Note for UoE users

If the installation is run through SSH, make sure you are in an *actual* machine and not just hare or bruegel, as these are just gateways and won't have a C compiler installed.