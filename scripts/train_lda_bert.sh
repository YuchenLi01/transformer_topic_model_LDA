#!/bin/bash

for hiddenlayers in 1
do
  for num_heads in 1
  do
    for lr in 0.01
    do
      python3 src/run_lm.py "config/bert_lda_hiddenlayers"$hiddenlayers"_heads"$num_heads"_lr"$lr"_one_hot.yaml"
    done
  done
done
