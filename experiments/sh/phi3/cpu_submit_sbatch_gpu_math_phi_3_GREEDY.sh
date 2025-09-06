#!/bin/zsh
#SBATCH --job-name=mctsr_phi3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=120G
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/test_mctsr_NE_OS_model_phi_3_mini_GREEDY.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/mctsr_NE_OS_model_phi_3_mini_GREEDY.py