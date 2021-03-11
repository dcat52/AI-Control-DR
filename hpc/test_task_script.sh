#!/bin/bash

# PLEASE USE THIS SCRIPT WHEN TESTING / TRYING TO UNDERSTAND THE SYSTEM!
# IT JUST SLEEPS FOR 5-10 SECONDS THEN EXITS. NICE AND SIMPLE / FAST!

# This script outputs some useful information so we can see what parallel
# and srun are doing.
SECONDS=0
sleepsecs=$[ ( $RANDOM % 10 ) + 5 ]s

# $1 is arg1:{1} from GNU parallel.
#
# $PARALLEL_SEQ is a special variable from GNU parallel. It gives the
# number of the job in the sequence.
#
# Here we print the sleep time, host name, and the date and time.
echo task $1 seq:$PARALLEL_SEQ sleep:$sleepsecs host:$(hostname) date:$(date)
echo args:$2

echo "Beginning Sleep. Zzzzz."
# Sleep a random amount of time.
sleep $sleepsecs
echo "Ahh, well rested!"
echo "Time Running (s): $SECONDS for job $1"