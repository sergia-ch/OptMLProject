#!/bin/bash
pids=""
python ../experiment.py --optimizer sgd --train_batch_size 1000 --learning_rate 1.0 --epochs 100 --image_side 4 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 1 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --p 3.0 --R 100.0 --gamma -1.0 --ro -1.0 --epochs 100 --image_side 4 --giveup 100 --accuracy_threshold 0.0 --repetitions 1 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer adam --train_batch_size 1000 --p 3.0 --learning_rate 0.001 --beta1 0.9 --beta2 0.999 --epsilon 1e-08 --epochs 100 --image_side 4 --giveup 100 --accuracy_threshold 0.0 --repetitions 1 --architecture 20_10 &
pids="$pids $!"
sleep 5
