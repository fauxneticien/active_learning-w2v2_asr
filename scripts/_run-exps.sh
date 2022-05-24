#!/bin/bash

# Run script from project-level directory (i.e. where scripts/, data/, etc. folders are)
# All path references are relative to project-level directory

USAGE="

usage: scripts/_run-exps.sh DATASET MODEL --acquisition

Examples:

scripts/_run-exps.sh gos-kdl facebook/wav2vec2-large random
"

DATASET=${1:?"Error. You must supply a dataset name. ${USAGE}"}
MODEL=${2:?"Error. You must supply a model repo or path. ${USAGE}"}
ACQ=${3:?"Error. You must supply an acquisition function. ${USAGE}"}

for i in {1..5}
do
   echo "Running iteration $i ..."

   # Make directory for outputs
   MODEL_NM=${MODEL##*/}
   WORKDIR=$"checkpoints/$DATASET/$MODEL_NM/$ACQ"

   # Pretend the TSV with all the (labelled) training data
   # is an 'Unlabelled pool (UPOOL)' from which we can ask for
   # subsets of data to be labelled by a human and returned to us
   UPOOL_TSV=$"data/datasets/$DATASET/datasets/train.tsv"

   DEV_TSV=$"data/datasets/$DATASET/datasets/dev.tsv"

   # Use echo to first test the commands are those you're expecting...
   mkdir -p $WORKDIR/$i

   if [ $i -eq 1 ]
   then
      # Model needs to be 'warm-started' before fine-tuning
      # Randomly sample 10% of data for first round
      echo "Warm-start round: acquiring data from '$UPOOL_TSV' using 'random' as the acquisition function"
      
      echo "python3 scripts/acquire.py random $UPOOL_TSV 0.10 $WORKDIR/$i/train-$i.tsv --seed $i"
      python3 scripts/acquire.py random $UPOOL_TSV 0.10 $WORKDIR/$i/train-$i.tsv --seed $i

   else
      # Active learning rounds (2, 3, 4, 5)
      echo "Acquiring data from '$UPOOL_TSV' using '$ACQ' as the acquisition function"

      # Pass in --lpool_tsv containing the data we've acquired so far
      I_MINUS_1=$(expr $i - 1)
      echo "python3 scripts/acquire.py $ACQ $UPOOL_TSV 0.10 $WORKDIR/$i/train-$i.tsv --lpool_tsv $WORKDIR/$I_MINUS_1/train-$I_MINUS_1.tsv --seed $i --checkpoint $WORKDIR/$I_MINUS_1"
      python3 scripts/acquire.py $ACQ $UPOOL_TSV 0.10 $WORKDIR/$i/train-$i.tsv --lpool_tsv $WORKDIR/$I_MINUS_1/train-$I_MINUS_1.tsv --seed $i --checkpoint $WORKDIR/$I_MINUS_1

   fi

   echo "python3 scripts/train_asr-by-w2v2-ft.py $MODEL $WORKDIR/$i $WORKDIR/$i/train-$i.tsv $DEV_TSV"
   python3 scripts/train_asr-by-w2v2-ft.py $MODEL $WORKDIR/$i $WORKDIR/$i/train-$i.tsv $DEV_TSV

   echo "python3 scripts/clean-checkpoints.py $WORKDIR/$i"
   python3 scripts/clean-checkpoints.py $WORKDIR/$i

   echo "---"
done
