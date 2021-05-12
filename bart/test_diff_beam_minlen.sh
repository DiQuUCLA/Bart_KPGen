#!/bin/sh

for BEAM in 1 2 3 4 5 6
do
    for MIN_LEN in 8 10 12 14 16
    do
        sh generate_output.sh 1,2 generation $BEAM $MIN_LEN
    done
done
