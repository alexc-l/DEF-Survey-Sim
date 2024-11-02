#!/usr/bin/env bash
#DSUB -n AECE_US_survey
#DSUB -N 1
#DSUB -A root.wuhkjdxyywyzstpu
#DSUB -R "cpu=32;gpu=1;mem=60000"
#DSUB -oo %J.out
#DSUB -eo %J.err

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/10.3.1
module load compilers/cuda/11.6.0
module load libs/cudnn/8.4.0.27_cuda11.x
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh
conda activate bits_introse

#export CUDA_VISIBLE_DEVICES=1

export prompt_path="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/survey_data/XLT_prompt_freeform.txt"
export questionnaire="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/survey_data/AECE/"
#export model_name="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/team/cache_dir/chatglm"
#export model_name="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/introse/cache/baichuan2-13B-4bits"
#export model_name="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/cache_dir/dolphin-2.2.1-mistral-7b"
export model_name="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/cache_dir/WizardLM-13B-V1.2"
export lang="English"
#export model_class="AutoModelForSeq2SeqLM"
export model_class="AutoModelForCausalLM"
export dtype="fp32"
#export generate_kwargs="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/config/generation_kwargs.json"
export generate_kwargs="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/config/generation_kwargs_causallm.json"
export case_num=20
export diverse_plan="all"
export output_dir="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/outputs/US_survey_all"
export test_per_case=1

python main_test.py \
  --prompt_path $prompt_path \
  --questionnaire $questionnaire \
  --model_name $model_name \
  --model_class $model_class \
  --dtype $dtype \
  --output_dir $output_dir \
  --generate_kwargs $generate_kwargs \
  --case_num $case_num \
  --diverse_plan $diverse_plan \
  --lang $lang \
  --test_per_case $test_per_case