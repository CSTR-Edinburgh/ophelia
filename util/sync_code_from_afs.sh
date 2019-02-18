#!/bin/bash

SERVERDIR=/group/project/cstr1/jasonfong/dc_tts_osw   ## group project space
SERVERDIR2=/disk/scratch_ssd/jasonfong/dc_tts_osw   ## adamski + hynek
SERVERDIR3=/disk/scratch_big/jasonfong/dc_tts_osw   ## lazar

SOURCE=/afs/inf.ed.ac.uk/user/s17/s1785140/dc_tts_osw  ## e.g. in your AFS home directory

rsync --exclude='.git/' -avzh --progress $SOURCE/ $SERVERDIR/ 
rsync --exclude='.git/' -avzh --progress $SOURCE/ $SERVERDIR2/
rsync --exclude='.git/' -avzh --progress $SOURCE/ $SERVERDIR3/
#  --exclude="*.txt"
