#!/usr/bin/env bash
#DSUB -n AECE_US_survey
#DSUB -N 1
#DSUB -A root.wuhkjdxyywyzstpu
#DSUB -R "cpu=32;gpu=1;mem=60000"
#DSUB -oo %J.out
#DSUB -eo %J.err

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles
module load compilers/cuda/11.6.0
module load tools/ccache/4.8.3
module load mpi/openmpi/4.1.5_cuda11.6
module load compilers/cuda/11.6.0
module load tools/simde/0.7.6
module load libs/openblas/0.3.25_gcc9.3.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate autogpt
#export CUDA_VISIBLE_DEVICES=3
#export CUDA_LAUNCH_BLOCKING=1
#export TORCH_USE_CUDA_DSA=true

export prompt_path="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/survey_data/XLT_prompt_freeform.txt"
export questionnaire="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/survey_data/AECE/"
export model_name="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/cache_dir/Mixtral-8x7B-Instruct-v0.1-bnb-4bit"
export model_name_short="mistral"
export lang="Chinese"
export model_class="AutoModelForCausalLM"
export dtype="fp32"
export generate_kwargs="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/config/generation_kwargs_causallm.json"
export case_num=20
export diverse_plan="all"
export output_dir="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/outputs/Mixtral-8x7B-Instruct-v0.1-bnb-4bit/"$lang
export test_per_case=1

export resume_epoch=0
export resume_input=0

python main_test.py \
  --prompt_path $prompt_path \
  --questionnaire $questionnaire \
  --model_name $model_name \
  --model_name_short $model_name_short \
  --model_class $model_class \
  --dtype $dtype \
  --output_dir $output_dir \
  --generate_kwargs $generate_kwargs \
  --case_num $case_num \
  --diverse_plan $diverse_plan \
  --lang $lang \
  --test_per_case $test_per_case \
  --resume_epoch $resume_epoch \
  --resume_input $resume_input