This repo is based on Wasi's NeuralKPGen repo. 

During modification, to make sure the code work correctly,
some code might be redudant.

Data under data/ directory and model under bart directory

Inside data directory, the generation folder is used to store and 
process data for inferencing only. You may run the inferencing like following:

    cd data/generation/
    sh run_generation.sh <TARGET>.jsonl
    cd ../../bart
    sh preprocess_generation.sh generation
    sh generate_output.sh GPU_NUMBER generation MODEL_DIR BEAM_SIZE MIN_LEN

for example:

    cd data/generation/
    sh run_generation.sh test.jsonl
    cd ../../bart
    sh preprocess_generation.sh generation
    sh generation_output.sh 1,2 generation /local/diq/kptimes 1 16

the output file will be in bart/logs directory with name output_test_1_16.hypo

For training the process is similar, you just change the dir to data/self_train
And for training you need to prepare three files name: train.jsonl, valid.jsonl, and test.jsonl and no need to specify files for preprocessing anymore
for example

    cd data/self_train/
    sh run.sh
    cd ../../bart
    sh preprocess_train.sh
    sh run.sh 1,2 self_train bart.base/model.pt /local/diq/self_train_checkpoints 10 4

The third parameter is the model you trained from, the forth one is where 
you want the trained model to be save
Notice that the parameters for each script are little different
The last two numbers are epochs and batch size
I've include a file called covid_test.jsonl as a sample for you to create your own test file
