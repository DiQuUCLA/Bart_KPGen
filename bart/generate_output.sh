#!/usr/bin/env bash

SRCDIR=data
mkdir -p logs

function train () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
BEAM_SIZE=$3
MIN_LEN=$4
TOTAL_NUM_UPDATES=40000
WARMUP_UPDATES=1000
LR=3e-05
MAX_TOKENS=4096
UPDATE_FREQ=4
ARCH=bart_base # bart_large
BART_PATH=bart.base/model.pt # bart.large/model.pt
SAVE_DIR=output_checkpoints

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
--required-batch-size-multiple 1 \
--arch $ARCH \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--dropout 0.1 --attention-dropout 0.1 \
--weight-decay 0.01 --optimizer adam --adam-betas "(0.9, 0.999)" --adam-eps 1e-08 \
--clip-norm 0.1 \
--lr-scheduler polynomial_decay --lr $LR \
--max-update $TOTAL_NUM_UPDATES --warmup-updates $WARMUP_UPDATES \
--update-freq $UPDATE_FREQ \
--skip-invalid-size-inputs-valid-test \
--find-unused-parameters --ddp-backend=no_c10d \
--save-dir $SAVE_DIR 2>&1 | tee $SAVE_DIR/output.log;

}

function decode () {

export CUDA_VISIBLE_DEVICES=$1
DATASET=$2
SAVE_DIR_PREFIX=$3
BEAM_SIZE=$4
MIN_LEN=$5

python decode.py \
--data_name_or_path "$SRCDIR/${DATASET}-bin/" \
--data_dir "$SRCDIR/${DATASET}/" \
--checkpoint_dir ${SAVE_DIR_PREFIX} \
--checkpoint_file checkpoint_best.pt \
--output_file logs/output_test_"$BEAM_SIZE"_"$MIN_LEN".hypo \
--batch_size 64 \
--beam  $BEAM_SIZE \
--min_len $MIN_LEN \
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

decode "$1" $2 $3 $4 $5
evaluate ${SRCDIR}/${2} "logs/taboola_test_"$4"_"$5".hypo" $2
