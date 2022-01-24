#!/bin/bash
#SBATCH --job-name="pi_2"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=pi_%j.out
#SBATCH --error=pi_%j.err

module purge
module load 2021
module load GCC/10.3.0

echo "OpenMP parallelism"
echo

export OMP_NUM_THREADS=16

for niter in 31250000 62500000 125000000 250000000 500000000 1000000000 2000000000
do
  export PI_N_ITER=$niter

  echo "nIter " $niter
  time ./pi
  echo "DONE "
done
