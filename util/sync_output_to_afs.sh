#!/bin/bash

PATTERN=$1

if [ `hostname` == hynek.inf.ed.ac.uk ] ; then 
    SOURCE=/disk/scratch_ssd/oliver/dc_tts/ ;
else
    echo 'do not know server: '
    echo `hostname`
    exit 1
fi

DEST=/afs/inf.ed.ac.uk/group/cstr/place/to/put/output/


HERE=`pwd`

cd $SOURCE

rsync -av --relative ./work/*${PATTERN}*/train-1/alignment* $DEST
rsync -av --relative ./work/*${PATTERN}*/synth/*.wav $DEST

cd $HERE

echo 'Synced'