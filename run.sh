#!/bin/bash

name="debug"
dim=4
sample="synthetic"
timelimit=3600
ord="1"

python main.py --sample=${sample} --dim=${dim} --sample=${sample} --timelimit=${timelimit} --ord=${ord}

