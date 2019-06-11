

mkdir /disk/scratch/oliver/temp/benchmark_data
cd /disk/scratch/oliver/temp/benchmark_data

ARCTIC_DB_DIR=http://tts.speech.cs.cmu.edu/awb/cmu_arctic
VNAME=slt
wget ${ARCTIC_DB_DIR}/cmu_us_${VNAME}_arctic.tar.bz2 &&
tar jxvf cmu_us_${VNAME}_arctic.tar.bz2
rm cmu_us_${VNAME}_arctic.tar.bz2


mkdir benchmark_slt_100
mkdir benchmark_slt_100/wav/

mv cmu_us_slt_arctic/wav/arctic_a00* benchmark_slt_100/wav/


awk '{print $2}' cmu_us_slt_arctic/etc/txt.done.data > names.txt
awk -F\" '{print $2}' cmu_us_slt_arctic/etc/txt.done.data > ./text.txt

paste -d\| names.txt text.txt text.txt > ./transcript_all.csv

head -99 ./transcript_all.csv > ./benchmark_slt_100/transcript_train.csv
tail -1 ./transcript_all.csv > ./benchmark_slt_100/transcript_test.csv

zip -r ./benchmark_slt_100.zip ./benchmark_slt_100/*

## copied the zip to dropbox https://www.dropbox.com/s/9lg0rou50gklx57/benchmark_slt_100.zip