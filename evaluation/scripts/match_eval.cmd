set lang=Chinese
set output_dir=outputs\%lang%_chatglm
set data_storage=human_eval\qa_score.csv
set questionnaire=survey_data\WVS7\
set instruction_path=survey_data\WVS7\human_eval_instructions.txt

python human_eval.py --lang %lang% --output_dir %output_dir% --data_storage %data_storage% --questionnaire %questionnaire% --instruction_path %instruction_path%