#!/bin/bash

PATTERN=$1

if [ `hostname` == hynek.inf.ed.ac.uk ] ; then 
    SOURCE=/disk/scratch_ssd/oliver/dc_tts_osw/ ;
elif [ `hostname` == stardale.inf.ed.ac.uk ] ; then 
    SOURCE=/disk/scratch/oliver/dc_tts_osw/ ;
elif [ `hostname` == starleader.inf.ed.ac.uk ] ; then 
    SOURCE=/disk/scratch/oliver/dc_tts_osw/ ;
else
    echo 'do not know server: '
    echo `hostname`
    exit 1
fi

DEST=/afs/inf.ed.ac.uk/group/cstr/projects/scar/SCRIPT/dc_tts_osw/output/


HERE=`pwd`

cd $SOURCE

# rsync -av --relative ./work/models/${PATTERN}*/train/model_epoch_*_devsynth*/*.wav $DEST
# rsync -av --relative ./work/models/${PATTERN}*/train/model_epoch_*_testsynth*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/train-1/alignment* $DEST
rsync -av --relative ./work/*${PATTERN}*/train-*/archive/*_devsynth/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/train-*/archive/*_devsynth/*/*
rsync -av --relative ./work/*${PATTERN}*/train-*/archive/*_synthesis/*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/train-*/archive/*_validation/*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/train-*/archive/*_synthesis/*/*.npy $DEST
# rsync -av --relative ./work/models/${PATTERN}*/train/*.npy $DEST       ## debugging files 
# rsync -av --relative ./work/models/${PATTERN}*/synthesis/epoch_*/*_trace.hdf $DEST
rsync -av --relative ./work/*${PATTERN}*/synth/*.wav $DEST

# e.g.:  /disk/scratch/oliver/convolin/work/models/test_renderer_density_01I/synthesis/renderer-49/00001/hvd_181.wav

cd $HERE


echo 'Synced to: /afs/inf.ed.ac.uk/group/cstr/projects/scar/SCRIPT/dc_tts/output/work'