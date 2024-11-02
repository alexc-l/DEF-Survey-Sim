#!/usr/bin/env bash
#DSUB -n EMNLP
#DSUB -N 1
#DSUB -A root.wuhkjdxyywyzstpu
#DSUB -R "cpu=32;gpu=1;mem=60000"
#DSUB -oo %J.out
#DSUB -eo %J.err

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.20_gcc10.5.0
module load compilers/gcc/10.5.0
module load compilers/cuda/11.8.0
module load libs/cudnn/8.4.0.27_cuda11.x
module load libs/nccl/2.17.1-1_cuda11.0

source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

conda activate emnlp_3.10

export CUDA_VISIBLE_DEVICES=2

export lang="Chinese"
export output_dir="outputs/Mixtral-8x7B-Instruct-v0.1-bnb-4bit/"$lang"/mistral"
export data_storage="gpt_judge/Mixtral-8x7B-Instruct-v0.1-bnb-4bit/"$lang
export questionnaire="survey_data/AECE_v1/"
export instruction_path="survey_data/AECE/CN_intellect_instructions.txt"
export intellect="yes"


python intellect_eval.py \
  --lang $lang \
  --output_dir $output_dir \
  --data_storage $data_storage \
  --questionnaire $questionnaire \
  --instruction_path $instruction_path \
  --intellect $intellect