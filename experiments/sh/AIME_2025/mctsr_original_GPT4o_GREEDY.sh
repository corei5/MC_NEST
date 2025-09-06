#!/bin/zsh
#SBATCH --job-name=NEG_gpt4o
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G

#SBATCH --partition=p_48G  # Partition for GPUs with >48GB memory


#SBATCH --output=/nfs/home/rabbyg/CAG/log/AIME_2025/mctsr_DUMMY_ANSWER_original_gpt5_PAIRWISE_IMPORTANCE_SAMPLING.log
source /nfs/home/rabbyg/.venv/bin/activate && python /nfs/home/rabbyg/CAG/AIME_dataset_exp/py_file/gpt4o/mctsr_original_GPT4o_GREEDY.py
