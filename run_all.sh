#!/bin/bash
python experiment.py --eta 0.1 --rho 0.0 --mu 0 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0.25 --mu 0 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0.5 --mu 0 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0.75 --mu 0 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 1.0 --mu 0 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0 --mu 0.0 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0 --mu 0.25 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0 --mu 0.5 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0 --mu 0.75 --epochs 20 --train_batch_size 256 &
python experiment.py --eta 0.1 --rho 0 --mu 1.0 --epochs 20 --train_batch_size 256 &

