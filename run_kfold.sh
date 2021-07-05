#!/bin/bash

set -e

# fold_id=0 busy for test fold
for FOLD_ID in 0 3 7; do
  for SEED_ID in 0 3; do
    python main.py "$@" --fold_id $FOLD_ID --random_seed_id $SEED_ID
  done
done 