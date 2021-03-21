#!/bin/bash

SECONDS=0

# Here we print the sleep time, host name, and the date and time.
echo task: $1 seq: $PARALLEL_SEQ host: $(hostname) date: $(date)
echo args: $2
echo store loc: $3

module load singularity

singularity exec ./hpc/output.sif python main.py \
    --train --episodes 1200 --save_prefix $3/$1/data \
    --print_freq 50 --save_freq 200 --write_freq 5 \
    --seed 539 \
    --std 0.10 --theta 0.16 \
    --update_freq 8 --tau 0.40 \
    --gamma 0.99 $2

echo "Time Running (s): $SECONDS for task $1"