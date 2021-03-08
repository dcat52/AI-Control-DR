#!/bin/bash
#SBATCH -N 1
#SBATCH -n 32
#SBATCH --array=1,2,3,4,5,10,20,30,40,50,60,70,80,90,100,150,200,250,300,350,400,450,500%10
#SBATCH --mem 16G
#SBATCH -p short 
#SBATCH -t 06:00:00
#SBATCH --output=./runs/%A_%a_std.out
#SBATCH --error=./runs/%A_%a_std.err
#SBATCH --mail-user dscatherman@wpi.edu
#SBATCH --mail-type=end

time=`date +%Y.%m.%d__%H.%M`
time=`date +%Y.%m.%d`
TASKID=$(printf "%03d" ${SLURM_ARRAY_TASK_ID})
store_dir=./runs/${time}__${SLURM_ARRAY_JOB_ID}
mkdir ${store_dir}
mkdir ${store_dir}/${TASKID}_weights

module load singularity

sstat -j $SLURM_ARRAY_JOB_ID.batch --format=JobID%12,MaxVMSize%8,MaxRSS%8,AveCPU%8
echo "====================="
echo "====================="
SECONDS=0
time singularity exec hpc/output.sif python main.py --train \
    --save_prefix ${store_dir}/${TASKID} \
    --save_freq 100 --print_freq 25 --write_freq 5 \
    --episodes 500 --update_freq ${TASKID}
echo "---------------------"
echo "Time running:"
echo $SECONDS
echo "====================="
echo "====================="
sstat -j $SLURM_ARRAY_JOB_ID.batch --format=JobID%12,MaxVMSize%8,MaxRSS%8,AveCPU%8
mv ./runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_std.out ${store_dir}/${TASKID}_std.out
mv ./runs/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_std.err ${store_dir}/${TASKID}_std.err
