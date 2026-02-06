#!/bin/bash
#SBATCH --job-name=blp
#SBATCH --output=prints/blp_%j.out
#SBATCH --error=prints/blp_%j.err
#SBATCH -p sched_mit_sloan_interactive
#SBATCH --time=0-1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=20G              
#SBATCH --cpus-per-task=24      # allow multithreaded numpy/pandas/TensorFlow to use n cores

# make conda available in the batch shell
source /home/software/anaconda3/2023.07/etc/profile.d/conda.sh

# activate your conda env
conda activate evio_env
#conda activate /home/heeney/evio_env

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK

# run your code
cd /home/heeney/evio_code/estimation/Dec17/
python blp_run.py
