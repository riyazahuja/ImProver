#!/bin/bash

#SBATCH --job-name=rerun_decl
#SBATCH --output=logs/final/rerun_decl.out
#SBATCH --error=logs/final/rerun_decl.err
#SBATCH --cpus-per-task=64
#SBATCH --partition=cpu
#SBATCH --qos=cpu_qos
#SBATCH --time=1-00:00:00
#SBATCH --mem=100G





source $HOME/miniconda/bin/activate env

cd /home/$USER/ImProver
lake build eval_improver
sleep 0.5


./improver run eval --run_id base_declarativity2_train --cpus 64
./improver run analysis --run_id base_declarativity2_train
