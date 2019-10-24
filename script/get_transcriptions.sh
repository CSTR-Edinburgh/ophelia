
python ./script/festival/csv2scm.py -i $1 -o utts.data

FEST='/afs/inf.ed.ac.uk/user/s15/s1520337/Documents/festival/festival/bin/festival'

SCRIPT=./script/festival/make_rich_phones_cmulex.scm
$FEST -b $SCRIPT | grep ___KEEP___ | sed 's/___KEEP___//' | tee ./transcript_temp1.csv

python $CODEDIR/script/festival/fix_transcript.py ./transcript_temp1.csv > ./new_transcript.csv
