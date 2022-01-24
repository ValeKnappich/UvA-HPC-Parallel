#!/bin/bash
#SBATCH --job-name="pi"
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

export PI_N_ITER=1000000000

for ncores in {1..32}

do
  export OMP_NUM_THREADS=$ncores

  echo "CPUS: " $OMP_NUM_THREADS
  time ./pi
  echo "DONE "
done
