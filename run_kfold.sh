#!/bin/bash

set -e

# fold_id=0 busy for test fold
for FOLD_ID in 0 1 2 3 4 5 6 7 8 9; do
  for SEED_ID in 0 1 2 3 4; do
    python -u main.py "$@" --fold_id $FOLD_ID --random_seed_id $SEED_ID -c $SEED_ID'_'$FOLD_ID'jarek'
  done
done