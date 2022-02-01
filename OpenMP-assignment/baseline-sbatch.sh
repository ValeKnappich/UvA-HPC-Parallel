#!/bin/bash -e
#SBATCH -t 03:00 -N 1 --constraint gold_6130 -o out/benchmark_%j.out

export BENCHMARK_N_ITERATIONS=10
export BENCHMARK_N=100000

./bin/baseline