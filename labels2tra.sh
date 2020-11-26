#!/bin/sh
#
# Replace phone sequence of transcript file with phone sequence from HTS style labels
# Usage: ./labels2tra.sh labels_dir transcript_file new_transcript_file

labelsdir=$1
trafile=$2
newtrafile=$3

cat $trafile | while IFS=$'|' read -r file nada text ps
do 

	grep -r "\[2\]" $labelsdir/$file.lab | sed 's/\+.*//' | sed 's/.*-//' > ~/tmp/test.txt

	newps=`cat ~/tmp/test.txt  | tr '\n' ' '`
        
	# To print start and end
    echo $file"||"$text"|<_START_> "${newps:4:-4}"<_END_>"

done > $newtrafile
