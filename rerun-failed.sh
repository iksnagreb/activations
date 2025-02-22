#!/bin/bash

# Find all "Failed" experiments from the dvc queue and extract their hashes
for exp in $(dvc queue status | grep Failed | cut -d' ' -f1); do
  # Apply the state of the failed experiment to the workspace and put it back
  # into the experiment queue forcing execution without cache
  dvc exp apply "$exp"; dvc exp run --queue --force;
done
