#!/bin/zsh
#SBATCH --job-name=mcIS_ori_phi3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G
#SBATCH --gres=gpu:a5000:2
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/mctsr_original_OS_model_phi_3_mini_IMPORTANCE_SAMPLING.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/py_file/phi_3_mini/mctsr_original_OS_model_phi_3_mini_IMPORTANCE_SAMPLING.py