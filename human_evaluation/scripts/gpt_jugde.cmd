set lang=Chinese
set output_dir=outputs\team_chatglm
set data_storage=gpt_judge\team_chatglm
set questionnaire=survey_data\WVS7\
set instruction_path=survey_data\AECE\CN_intellect_instructions.txt
set intellect=yes

python intellect_eval.py --lang %lang% --output_dir %output_dir% --data_storage %data_storage% --questionnaire %questionnaire% --instruction_path %instruction_path% --intellect %intellect%