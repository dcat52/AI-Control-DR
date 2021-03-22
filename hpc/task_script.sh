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
    --std 0.10 --theta 0.10 \
    --update_freq 2 --tau 0.015 \
    --batch_size 32 --buffer_capacity 10000 \
    --actor_lr 0.0001 --critic_lr 0.0002 \
    --gamma 0.99 $2

echo "Time Running (s): $SECONDS for task $1"
