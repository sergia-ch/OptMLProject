python experiment.py --optimizer sgd --train_batch_size 1000 --learning_rate 0.1 --epochs 100 --image_side 10 --giveup 100 --accuracy_threshold 0 --repetitions 1 &
python experiment.py --optimizer frankwolfe --train_batch_size 1000 --R 100 --p 2 --gamma 0.01 --ro 0.6 --epochs 200 --image_side 10 --giveup 100 --accuracy_threshold 0 --repetitions 1 &
python experiment.py --optimizer adam --train_batch_size 1000 --learning_rate 0.0 --beta1 0.9 --beta2 1.0 --epsilon 0.0 --epochs 200 --image_side 10 --giveup 100 --accuracy_threshold 0 --repetitions 1 &

