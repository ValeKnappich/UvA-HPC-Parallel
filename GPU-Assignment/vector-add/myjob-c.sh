#!/bin/bash
#SBATCH -t 5:00
#SBATCH -n 1 
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_shared_course
#SBATCH --mem=100000M

export VECTOR_ADD_N=655360

for blocksize in 64 128 256 512 1024
do
    export VECTOR_ADD_BLOCK_SIZE=$blocksize
    ./vector-add
    echo
done