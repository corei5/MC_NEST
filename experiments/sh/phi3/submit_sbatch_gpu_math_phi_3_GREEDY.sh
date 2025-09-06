#!/bin/zsh
#SBATCH --job-name=mctsr_phi3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:t2080ti:1
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/test_mctsr_NE_OS_model_phi_3_mini_GREEDY.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/mctsr_NE_OS_model_phi_3_mini_GREEDY.py