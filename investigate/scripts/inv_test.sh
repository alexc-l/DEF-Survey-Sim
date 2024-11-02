#!/usr/bin/env bash
#DSUB -n xlsum_pretrain
#DSUB -N 1
#DSUB -A root.wuhkjdxjsjkxyjsxyuan
#DSUB -R "cpu=128;gpu=4;mem=240000"
#DSUB -oo %J.out
#DSUB -eo %J.err

source /home/HPCBase/tools/module-5.2.0/init/profile.sh
module purge
module use /home/HPCBase/modulefiles/
module load libs/openblas/0.3.18_kgcc9.3.1
module load compilers/gcc/9.3.0
module load compilers/cuda/11.3.0
module load libs/cudnn/8.2.1_cuda11.3
module load libs/nccl/2.17.1-1_cuda11.0
source /home/HPCBase/tools/anaconda3/etc/profile.d/conda.sh

conda activate llmvi

export CUDA_VISIBLE_DEVICES=0

export prompt_path="D:\\projects\\LLMValueInvestigation\\investigate\\survey_data\\XLT_prompt.txt"
export questionnaire="D:\\projects\\LLMValueInvestigation\\investigate\\survey_data\\WVS7\\"
#export model_name="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/xlsum/seq2seq/cache_dir/chatbloom"
export model_name="gpt-3.5-turbo"
export model_class="AutoModelForCausalLM"
export dtype="bf16"
export deployment_framework="hf_accelerate"
export generate_kwargs="D:\\projects\\LLMValueInvestigation\\investigate\\config\\generation_kwargs.json"
export case_num=2
export output_dir="D:\\projects\\LLMValueInvestigation\\investigate\\outputs"

python main_test.py \
  --prompt_path $prompt_path \
  --questionnaire $questionnaire \
  --model_name $model_name \
  --model_class $model_class \
  --dtype $dtype \
  --output_dir $output_dir \
  --generate_kwargs $generate_kwargs \
  --case_num $case_num \
  --deployment_framework $deployment_framework