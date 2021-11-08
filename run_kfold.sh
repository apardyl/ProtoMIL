#!/bin/bash

set -e

# fold_id=0 busy for test fold
for FOLD_ID in 0; do
  for SEED_ID in 0 1; do
    python -u main.py "$@" --fold_id $FOLD_ID --random_seed_id $SEED_ID -c $SEED_ID'_'$FOLD_ID'bce_tree'
  done
done