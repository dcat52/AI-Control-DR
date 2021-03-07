#!/bin/bash
#SBATCH --output=./runs/%j_std.out 
#SBATCH --error=./runs/%j_std.err 
#SBATCH -N 1
#SBATCH -n 8
#SBATCH --mem 8G
#SBATCH -p short 
#SBATCH -t 01:00:00

time=`date +%Y-%m-%d_%H-%M`
store_dir=./runs/${time}_${SLURM_JOBID}
mkdir ${store_dir}

module load singularity

sstat -j $SLURM_JOB_ID.batch --format=JobID%12,MaxVMSize%8,MaxRSS%8,AveCPU%8
singularity exec hpc/output.sif python main.py --train --save_dir ${store_dir}/weights --save_freq 100 --episodes 100
sstat -j $SLURM_JOB_ID.batch --format=JobID%12,MaxVMSize%8,MaxRSS%8,AveCPU%8
mv ./runs/${SLURM_JOBID}_std.out ${store_dir}/std.out
mv ./runs/${SLURM_JOBID}_std.err ${store_dir}/std.err

