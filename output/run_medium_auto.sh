#!/bin/bash
pids=""
python ../experiment.py --optimizer sgd --train_batch_size 1000 --learning_rate 0.1 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer sgd --train_batch_size 1000 --learning_rate 0.01 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer sgd --train_batch_size 1000 --learning_rate 0.001 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer sgd --train_batch_size 1000 --learning_rate 0.0001 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
pids=""
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma -1.0 --ro -1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma -1.0 --ro 0.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma -1.0 --ro 0.5 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma -1.0 --ro 1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
pids=""
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.1 --ro -1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.1 --ro 0.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.1 --ro 0.5 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.1 --ro 1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
pids=""
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.5 --ro -1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.5 --ro 0.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.5 --ro 0.5 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 0.5 --ro 1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
pids=""
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 1.0 --ro -1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 1.0 --ro 0.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 1.0 --ro 0.5 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 20.0 --gamma 1.0 --ro 1.0 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
pids=""
python ../experiment.py --optimizer frankwolfe --train_batch_size 60000 --R 20.0 --gamma -1.0 --ro 1.0 --epochs 5000 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 60000 --R 20.0 --gamma 0.1 --ro 1.0 --epochs 5000 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 60000 --R 20.0 --gamma 0.3 --ro 1.0 --epochs 5000 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer frankwolfe --train_batch_size 60000 --R 20.0 --gamma 0.5 --ro 1.0 --epochs 5000 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
pids=""
python ../experiment.py --optimizer adam --train_batch_size 1000 --learning_rate 0.1 --beta1 0.9 --beta2 0.999 --epsilon 1e-08 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer adam --train_batch_size 1000 --learning_rate 0.01 --beta1 0.9 --beta2 0.999 --epsilon 1e-08 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer adam --train_batch_size 1000 --learning_rate 0.001 --beta1 0.9 --beta2 0.999 --epsilon 1e-08 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
sleep 5
python ../experiment.py --optimizer adam --train_batch_size 1000 --learning_rate 0.0001 --beta1 0.9 --beta2 0.999 --epsilon 1e-08 --epochs 100 --image_side 5 --giveup 100 --accuracy_threshold 0.0 --p 3.0 --repetitions 3 --architecture 20_10 &
pids="$pids $!"
wait $pids
sleep 5
