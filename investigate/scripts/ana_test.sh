#!/usr/bin/env bash

conda activate llmvi

export lang="English"
export output_dir="outputs/Meta-Llama-3-8B/Chinese/llama-3-base"
export data_storage="outputs/Meta-Llama-3-8B/Chinese/llama-3-base/qa_score.csv"
export questionnaire="/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/LLMVI/investigate/survey_data/AECE/"

python auto_analyze.py \
    --lang $lang \
    --output_dir $output_dir \
    --data_storage $data_storage \
    --questionnaire $questionnaire