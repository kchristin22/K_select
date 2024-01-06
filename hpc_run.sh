#!/bin/bash
#SBATCH --job-name=k_select
#SBATCH --partition=batch
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --time=3:00

module load gcc/12.2.0 openmpi curl/8.0.1-clqyiyn cmake

mkdir -p build
cd build/
cmake ..
cmake --build .
cd bin/
cp ../../sorted_data.txt .
srun -N $SLURM_JOB_NUM_NODES --ntasks-per-node $SLURM_NTASKS_PER_NODE ./output 14042179 # k = n / 2