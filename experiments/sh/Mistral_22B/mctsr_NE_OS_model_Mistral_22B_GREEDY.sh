#!/bin/zsh
#SBATCH --job-name=mis_NG
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=20G
#SBATCH --gres=gpu:l40s:1
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/Mistral_22B/mctsr_NE_OS_model_Mistral_22B_GREEDY.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/py_file/Mistral_22B/mctsr_NE_OS_model_Mistral_22B_GREEDY.py