#!/usr/bin/env bash


function download_data () {

declare -A FILEIDS
FILEIDS["train"]=1LGR62JPHL2-zesX5lT53KqfzHB78g0d_
FILEIDS["valid"]=1Fq3oZR99OTYKKe88-5ocwbdMT_CMdcEb
FILEIDS["test"]=1F-HDwjI23f6nvtFiea-CGIO_IaKi_2fK

URL_PREFIX=https://drive.google.com/uc?export=download
downloaded=false

for split in train valid test
do
    FILE=KPTimes.${split}.jsonl
    if [[ ! -f "$FILE" ]]; then
        downloaded=true
        FILEID=${FILEIDS[${split}]}
        curl -c ./cookie -s -L "${URL_PREFIX}&id=${FILEID}" > /dev/null
        curl -Lb ./cookie "${URL_PREFIX}&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${FILEID}" -o ${FILE}.gz
        gunzip ${FILE}.gz
        rm ./cookie
    fi
done

}

function prepare () {

SRC_DIR=../..
OUTDIR=$2

if [[ ! -d $OUTDIR ]]; then
    echo "============Processing generation dataset============"
    PYTHONPATH=$SRC_DIR python -W ignore prepare_generation.py \
        -dataset $3 \
        -data_dir . \
        -out_dir $OUTDIR \
        -tokenizer $1 \
        -workers 60
fi

}

echo $1
prepare BertTokenizer processed $1
prepare SpacyTokenizer spacy_processed $1


