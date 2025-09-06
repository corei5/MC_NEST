#!/bin/zsh
#SBATCH --job-name=mctsr_ori_phi3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:a3090:2
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/mctsr_original_OS_model_phi_3_mini_PAIRWISE_IMPORTANCE_SAMPLING.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/mctsr_original_OS_model_phi_3_mini_PAIRWISE_IMPORTANCE_SAMPLING.py