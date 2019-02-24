#!/bin/bash

DEST=~/samples_dctts  ## e.g. directory on AFS where examining audio is straightforward

PATTERN=$1  ## substring of the config to sync

rsync -av --relative ./work/*${PATTERN}*/train-t2m/validation_epoch_*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/train-ssrn/validation_epoch_*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/train-t2m/alignment* $DEST
rsync -av --relative ./work/*${PATTERN}*/synth/*/*.wav $DEST
rsync -av --relative ./work/*${PATTERN}*/synth/*/*.png $DEST


echo "Synced to: $DEST"