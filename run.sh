#!/bin/bash
#SBATCH --time=00:30:00
#SBATCH --account=ml4h_jobs
#SBATCH --mem=8G

exec > "run_${1}_${SLURM_JOB_ID}.out" 2>&1

module load cuda/12.6
source ~/ml4hc_env/bin/activate

cd ~/ml4hc-project2
PYTHONUNBUFFERED=1 python3 -m "part1.$1" "${@:2}"
