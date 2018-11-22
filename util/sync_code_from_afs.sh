#!/bin/bash

if [ `hostname` == hynek.inf.ed.ac.uk ] ; then 
    SERVERDIR=/disk/scratch_ssd/oliver/dc_tts/ ;
else
    echo 'do not know server: '
    echo `hostname`
    exit 1
fi

SOURCE=/afs/inf.ed.ac.uk/user/location_of/code

DEST=$SERVERDIR

rsync -avzh --progress $SOURCE/ $DEST/ 


