#!/bin/bash

python legged_gym/legged_gym/scripts/steady_state_evaluation.py \
  --eval-mode collect \
  --cpg-parameter-mode nominal \
  --output-dir steady_state_results \
  --episodes 30 \
  --warmup-s 3 \
  --measure-s 10 \
  --save-timeseries \
  --task silver_badger_rigid_spine \
  --load_run cerulean-energy-44 \
  --checkpoint 3000 \
  --headless
