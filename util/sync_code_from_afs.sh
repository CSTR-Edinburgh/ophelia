#!/bin/bash

if [ `hostname` == hynek.inf.ed.ac.uk ] ; then 
    SERVERDIR=/disk/scratch_ssd/oliver/dc_tts_osw/ ;
elif [ `hostname` == stardale.inf.ed.ac.uk ] ; then 
    SERVERDIR=/disk/scratch/oliver/dc_tts_osw/ ;
elif [ `hostname` == starleader.inf.ed.ac.uk ] ; then 
    SERVERDIR=/disk/scratch/oliver/dc_tts_osw/ ;
else
    echo 'do not know server: '
    echo `hostname`
    exit 1
fi


SOURCE=/afs/inf.ed.ac.uk/user/o/owatts/repos/dc_tts_osw

DEST=$SERVERDIR

# mkdir -p $OUTDIR/synched_from_hynek

rsync -avzh --progress $SOURCE/ $DEST/ 

#  --exclude="*.txt"
