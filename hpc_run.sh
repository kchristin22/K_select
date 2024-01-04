#!/bin/bash
#SBATCH --job-name=k_select
#SBATCH --partition=rome
#SBATCH --ntasks=8
#SBATCH --cpus-per-task=2
#SBATCH --time=10:00

module load gcc/12.2.0 openmpi curl/8.0.1-clqyiyn cmake

cd ../build
cmake ..
cmake --build .
cd ../bin
srun -n $SLURM_NTASKS ./output 397