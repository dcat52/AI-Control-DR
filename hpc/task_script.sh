#!/bin/bash

SECONDS=0

# Here we print the sleep time, host name, and the date and time.
echo task: $1 seq: $PARALLEL_SEQ host: $(hostname) date: $(date)
echo args: $2
echo store loc: $3

module load singularity

singularity exec ./hpc/output.sif python main.py \
    --train --episodes 400 --save_prefix $3/$1/data \
    --print_freq 600 --save_freq 600 --write_freq 1 \
    --tensorboard 0 \
    --seed 539 \
    --std 0.10 --theta 0.05 \
    --update_freq 2 --tau 0.05 \
    --batch_size 256 --buffer_capacity 10000 \
    --actor_lr 0.0001 --critic_lr 0.0002 \
    --gamma 0.99 $2

echo "Time Running (s): $SECONDS for task $1"
