#!/usr/bin/env bash

conda activate llmvi

export CUDA_VISIBLE_DEVICES=0
export TORCH_USE_CUDA_DSA=true

export prompt_path="survey_data/XLT_prompt_freeform.txt"
export questionnaire="survey_data/AECE/"
export model_name="investigate/cache_dir/Mixtral-8x7B-Instruct-v0.1-bnb-4bit"
export model_name_short="mistral"
export lang="Chinese"
export model_class="AutoModelForCausalLM"
export generate_kwargs="investigate/config/generation_kwargs_causallm.json"
export case_num=20
export diverse_plan="all"
export output_dir="investigate/outputs/Mixtral-8x7B-Instruct-v0.1-bnb-4bit/"$lang

export resume_epoch=0
export resume_input=0

python main_test.py \
  --prompt_path $prompt_path \
  --questionnaire $questionnaire \
  --model_name $model_name \
  --model_name_short $model_name_short \
  --model_class $model_class \
  --output_dir $output_dir \
  --generate_kwargs $generate_kwargs \
  --case_num $case_num \
  --diverse_plan $diverse_plan \
  --lang $lang \
  --resume_epoch $resume_epoch \
  --resume_input $resume_input