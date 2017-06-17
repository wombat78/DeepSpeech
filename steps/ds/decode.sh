#!/bin/bash


SPEECHROOT=/home/data/speech/eng/
TESTSET=wsj-test_dev93
TMPDIR=tmp/
RESULT=result.txt
SHOW_DECODING=True

#/home/data/speech/eng/wsj-test_dev93.scp

system_name="deepspeech/ted"
checkpoint_dir=

. path.sh
. utils/parse_options.sh

#  figure out where the reference files for Kaldi are
scpf=$SPEECHROOT/$TESTSET.scp
reff=$SPEECHROOT/$TESTSET.ref

# prepare .csv file readable by deepspeech
mytmpdir=$TMPDIR/$TESTSET

if [ ! -d $mytmpdir ]; then 
    mkdir -p $mytmpdir
    echo Converting kaldi data set for deepspeech in $mytmpdir

    cat $scpf | awk '{ print $1 " '$mytmpdir'/" $1 ".wav"; }' | sort -k 1,1 > $mytmpdir/out.scp
    wav-copy scp:$scpf scp:$mytmpdir/out.scp

    cat $mytmpdir/out.scp | awk '{ print "readlink -f " $2;}' | sh | awk '{ print "wc -c " $1;}' | sh | awk '{ print $2 "," $1 ",";}'  > $mytmpdir/paths.txt
    
    # NOTE, we heavily clean up the core code, so that only lowercase alphabets are left
    cat $reff | sort -k 1,1 | cut -d " " -f 2- | tr A-Z a-z |\
        perl -ne 'chomp; s/<[^>]+>/ /g; s/[^a-z ]/ /g; print "$_\n";' > $mytmpdir/transcripts.txt
    
    echo wav_filename,wav_filesize,transcript > $mytmpdir/$TESTSET.csv
    paste -d '' $mytmpdir/paths.txt $mytmpdir/transcripts.txt >> $mytmpdir/$TESTSET.csv
    echo Created DeepSpeech csv file in $mytmpdir/$TESTSET.csv
fi

if [ -d "${COMPUTE_KEEP_DIR}" ];then 
    checkpoint_dir=$COMPUTE_KEEP_DIR
else
    checkpoint_dir=$(python -c 'from xdg import BaseDirectory as xdg; print(xdg.save_data_path("'$system_name'"))')
fi

# run the decoder
echo Decoding $TESTSET with model in $checkpoint_dir
set -xe

python -u TestDeepSpeech.py \
     --result $RESULT \
     --log_level 0 \
     --show_decoding $SHOW_DECODING \
     --log_traffic True \
     --test_files "$mytmpdir/$TESTSET.csv" \
     --train_batch_size 16 \
     --dev_batch_size 8 \
     --test_batch_size 8 \
     --epoch 0 \
     --display_step 10 \
     --validation_step 1 \
     --dropout_rate 0.30 \
     --default_stddev 0.046875 \
     --learning_rate 0.0001 \
     --fulltrace True \
     --checkpoint_dir "$checkpoint_dir" \
     "$@" |& tee test.log



