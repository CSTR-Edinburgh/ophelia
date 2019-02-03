#!/bin/bash

DEST=~/synthetic_speech  ## e.g. directory on AFS where examining audio is straightforward

PATTERN=$1  ## substring of the config to sync

rsync -av --relative ./work/*${PATTERN}*/train-t2m/alignment* $DEST
rsync -av --relative ./work/*${PATTERN}*/synth/*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/synth/*/*.wav $DEST

echo "Synced to: $DEST"