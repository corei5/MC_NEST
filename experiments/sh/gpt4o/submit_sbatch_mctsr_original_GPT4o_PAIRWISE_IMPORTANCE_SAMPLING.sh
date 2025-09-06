#!/bin/zsh
#SBATCH --job-name=OPIS_gpt4o
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --output=/nfs/home/rabbyg/CAG/AIME_dataset_exp/log/gpt4o/mctsr_original_GPT4o_PAIRWISE_IMPORTANCE_SAMPLING.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/py_file/gpt4o/mctsr_original_GPT4o_PAIRWISE_IMPORTANCE_SAMPLING.py
