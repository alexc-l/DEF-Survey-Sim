set lang=Chinese
set output_dir=outputs\Mixtral-8x7B-Instruct-v0.1-bnb-4bit\Chinese\mistral
set data_storage=human_eval\Mixtral-8x7B-Instruct-v0.1-bnb-4bit\Chinese\mistral
set questionnaire=survey_data\AECE_v1\
set instruction_path=survey_data\AECE\CN_human_instructions.txt
set intellect=no

python human_eval.py --lang %lang% --output_dir %output_dir% --data_storage %data_storage% --questionnaire %questionnaire% --instruction_path %instruction_path% --intellect %intellect%