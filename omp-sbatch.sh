#!/bin/bash -e
#SBATCH -t 03:00 -N 1 --constraint gold_6130

export OMP_NUM_THREADS=`nproc --all`
export BENCHMARK_N_ITERATIONS=10
export BENCHMARK_N=100000

./omp
