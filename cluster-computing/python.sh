#!/bin/bash
#SBATCH --job-name="python"
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=10
#SBATCH --ntasks=10
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --output=python_%j.out
# #SBATCH --reservation=uva_course

module purge
module load 2019
module load Python/3.6.6-foss-2018b

echo "PYTHON EXAMPLE"
time python linalg.py 
echo "DONE "
