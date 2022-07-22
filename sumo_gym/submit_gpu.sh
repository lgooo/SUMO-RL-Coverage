#! /bin/bash
#
#SBATCH --partition gpu
#SBATCH --mem=8G
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -G 1
#SBATCH -c 1
#SBATCH --time 24:00:00

srun -N 1 -n 1 -G 1 --exclusive ./run.sh $@
