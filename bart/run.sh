#!/usr/bin/env bash

SRCDIR=data
mkdir -p logs
TOTAL_EPOCHS=$5
BATCH_SIZE=$6

function train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
LR=3e-05
MAX_TOKENS=4096
UPDATE_FREQ=4
ARCH=bart_base # bart_large
BART_PATH=$3 # bart.large/model.pt
SAVE_DIR=$4

fairseq-train ${SRCDIR}/${DATASET}-bin/ \
--restore-file $BART_PATH \
--max-tokens $MAX_TOKENS \
--task translation \
--truncate-source \
--max-source-positions 1024 --max-target-positions 1024 \
--source-lang source --target-lang target \
--layernorm-embedding \
--share-all-embeddings \
--share-decoder-input-output-embed \
--reset-optimizer --reset-dataloader --reset-meters \
--batch-size $BATCH_SIZE \
--arch $ARCH \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr-scheduler polynomial_decay --lr $LR \
--max-epoch $TOTAL_EPOCHS \
--update-freq $UPDATE_FREQ \
--skip-invalid-size-inputs-valid-test \
--find-unused-parameters --ddp-backend=no_c10d \
--save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/output.log;

}

function decode () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR_PREFIX=$3

python decode.py \
--data_name_or_path "$SRCDIR/${DATASET}-bin/" \
--data_dir "$SRCDIR/${DATASET}/" \
--checkpoint_dir $SAVE_DIR_PREFIX \
--checkpoint_file checkpoint_best.pt \
--output_file logs/${DATASET}_test.hypo \
--batch_size 64 \
--beam 1 \
--min_len 1 \
--lenpen 1.0 \
--no_repeat_ngram_size 3 \
--max_len_b 60;

}

function evaluate () {

python -W ignore kp_eval.py \
--src_dir $1 \
--pred_file $2 \
--tgt_dir . \
--log_file $3 \
--k_list 5 M;

}

while getopts ":h" option; do
   case $option in
      h) # display Help
        echo
        echo "Syntax: run.sh GPU_ID DATASET_NAME"
        echo
        echo "GPU_ID         A list of gpu ids, separated by comma. e.g., '0,1,2'"
        echo "DATASET_NAME   Name of the training dataset. e.g., kp20k, kptimes, etc."
        echo
        exit;;
   esac
done

if [[ $2 == 'kp20k' ]]; then
    train "$1" $2 
    for dataset in kp20k inspec krapivin nus semeval; do
        decode "$1" $dataset $2
        evaluate ${SRCDIR}/${dataset} "logs/${dataset}_test.hypo" $dataset
    done
elif [[ $2 == 'kptimes' ]]; then
    train "$1" $2 $3 $4
    decode "$1" $2 $4
    evaluate ${SRCDIR}/${2} "logs/${2}_test.hypo" $2
elif [[ $2 == 'self_train' ]]; then
    train "$1" $2 $3 $4
    decode "$1" $2 $4
    evaluate ${SRCDIR}/${2} "logs/${2}_test.hypo" $2
fi
