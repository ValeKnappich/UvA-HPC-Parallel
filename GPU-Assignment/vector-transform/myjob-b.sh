#!/bin/bash
#SBATCH -t 5:00
#SBATCH -n 1 
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_shared_course
#SBATCH --mem=100000M

for n in 256 1024 65536 655360 1000000
do
    export VECTOR_ADD_N=$n
    ./vector-transform
    echo
done