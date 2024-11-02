set lang=English
set output_dir=outputs\Mixtral-8x7B-Instruct-v0.1-bnb-4bit\English\mistral
set data_storage=intellect_eval\Mixtral-8x7B-Instruct-v0.1-bnb-4bit\English\mistral
set questionnaire=survey_data\AECE_v1\
set instruction_path=survey_data\AECE\US_intellect_instructions.txt
set intellect=yes

python human_eval.py --lang %lang% --output_dir %output_dir% --data_storage %data_storage% --questionnaire %questionnaire% --instruction_path %instruction_path% --intellect %intellect%