#!/bin/bash

# Download RawData
# mkdir -p ./data/ISRUC_S3/RawData && echo 'Make data dir: ./data/ISRUC_S3'

# cd ./data/ISRUC_S3/RawData
# for s in $(seq 1 10)
# do
#     wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/$s.rar
#     unrar x $s.rar
#     if [ -f "$s/$s.rec" ]; then
#         mv "$s/$s.rec" "$s/$s.edf"
#     fi
# done
# echo 'Download Data to "./data/ISRUC_S3/RawData" complete.'

# cd ../../../

# Download ExtractedChannels
mkdir -p ./data/ISRUC_S3/ExtractedChannels && echo 'Make data dir: ./data/ISRUC_S3/ExtractedChannels/'

cd ./data/ISRUC_S3/ExtractedChannels
for s in $(seq 4 10)
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupIII-Extractedchannels/subject$s.mat 
done
echo 'Download Data to "./data/ISRUC_S3/ExtractedChannels" complete.'
