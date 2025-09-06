#!/bin/zsh
#SBATCH --job-name=mctsr_pis
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/mctsr_NE_OS_model_phi_3_mini_PAIRWISE_IMPORTANCE_SAMPLING.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/mctsr_NE_OS_model_phi_3_mini_PAIRWISE_IMPORTANCE_SAMPLING.py