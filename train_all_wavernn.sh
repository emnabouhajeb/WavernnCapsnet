#!/bin/bash

echo "Starting WaveRNN pre-training for all macro-classes..."

python wavernn_pretrain.py --macroclass Vowels
python wavernn_pretrain.py --macroclass Stops
python wavernn_pretrain.py --macroclass Fricatives
python wavernn_pretrain.py --macroclass Nasals
python wavernn_pretrain.py --macroclass Affricates
python wavernn_pretrain.py --macroclass Semivowels
python wavernn_pretrain.py --macroclass Others

echo "All WaveRNN models trained!"