#!/bin/bash -e
#SBATCH -t 3:00 -N 1 --constraint gold_6130

export OMP_NUM_THREADS=`nproc --all`

./omp