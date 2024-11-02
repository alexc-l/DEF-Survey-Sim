#!/usr/bin/env bash

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/10.3.1
module load compilers/cuda/11.3.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

conda activate llmvi

export lang="Chinese"
export output_dir="outputs/Mixtral-8x7B-Instruct-v0.1-bnb-4bit/"$lang"/mistral"
export data_storage="outputs/Mixtral-8x7B-Instruct-v0.1-bnb-4bit/"$lang"/mistral/qa_score.csv"
export questionnaire="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/survey_data/AECE/"

python auto_analyze.py \
    --lang $lang \
    --output_dir $output_dir \
    --data_storage $data_storage \
    --questionnaire $questionnaire
