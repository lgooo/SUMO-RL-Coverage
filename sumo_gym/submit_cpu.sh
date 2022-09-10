#! /bin/bash
#
#SBATCH --mem=2G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 1
#SBATCH --time 24:00:00

srun -N 1 -n 1 -c 1 --exclusive ./run.sh $@
