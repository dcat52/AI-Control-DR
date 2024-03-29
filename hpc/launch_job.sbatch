#!/bin/bash
#SBATCH --output=./runs/slurm_%j_std.out
#SBATCH --error=./runs/slurm_%j_std.err
#SBATCH --partition=short
#SBATCH --mail-user dscatherman@wpi.edu
#SBATCH --mail-type=end
#SBATCH --mem-per-cpu=3G
#SBATCH --time=24:00:00
#SBATCH --nodes=4-8
#SBATCH --ntasks=384

time=`date +%Y.%m.%d__%H.%M.%S`
store_dir=./runs/${time}_$2
mkdir ${store_dir}

###SBATCH --nodes=7-8
###SBATCH --ntasks-per-node=24
###SBATCH --mincpus=48
###SBATCH --exclusive
###SBATCH --cpus-per-task=2
# Load the default version of GNU parallel.
module load parallel

echo "Using file $1"
srun="srun --exclusive -N1 -n1"
fake="echo {JN}"

pre="mkdir -p ${store_dir}/{JN} ; echo submitting job {JN}"
post="echo exited job {JN}"

TOTAL_NNODES=`expr $SLURM_NNODES \* $SLURM_NTASKS_PER_NODE`
echo "Running $TOTAL_NNODES jobs at once."

# many jobs should be run simultaneously.
parallel="parallel --header : --line-buffer --delay 0.2 -j $SLURM_NTASKS --joblog runs/job.log --retries 3"

# Run the parallel command.
SECONDS=0
$parallel --colsep ', ' "$pre; $srun ./hpc/task_script.sh {JN} {ARGS} ${store_dir} > ${store_dir}/{JN}/std.out; $post" :::: $1
echo "Time Running (s):" $SECONDS

mv runs/slurm_${SLURM_JOB_ID}_std.out $store_dir/std.out
mv runs/slurm_${SLURM_JOB_ID}_std.err $store_dir/std.err
mv runs/job.log $store_dir/job.log
