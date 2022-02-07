#!/bin/bash
#SBATCH -t 5:00
#SBATCH -n 1 
#SBATCH --gres=gpu:1 
#SBATCH --partition=gpu_shared_course
#SBATCH --mem=100000M
#SBATCH --output slurm-exercise-4-%j.out

export KEY=3
./crypto-exercise-4
