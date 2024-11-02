#!/usr/bin/env bash

export lang="Chinese"
export output_dir="outputs/dolphin-Llama/"$lang"/llama-3"
export data_storage="intellect_eval/dolphin-Llama/"$lang"/intellect_score.csv"
export questionnaire="survey_data/AECE/"
export instruction_path="survey_data/AECE/CN_intellect_instructions.txt"
export intellect="yes"

python human_eval.py \
  --lang $lang \
  --output_dir $output_dir \
  --data_storage $data_storage \
  --questionnaire $questionnaire \
  --instruction_path $instruction_path \
  --intellect $intellect