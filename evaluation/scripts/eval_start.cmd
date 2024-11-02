set lang=Chinese
set output_dir=outputs\introse\%lang%_chatglm
set data_storage=human_eval\introse\intellect_score.csv
set questionnaire=survey_data\introse\
set instruction_path=survey_data\introse\intellect_instructions.txt
set intellect=yes

python human_eval.py --lang %lang% --output_dir %output_dir% --data_storage %data_storage% --questionnaire %questionnaire% --instruction_path %instruction_path% --intellect %intellect%