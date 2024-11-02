set prompt_path=survey_data\XLT_prompt_freeform.txt
set questionnaire=survey_data\AECE\
set model_name=claude-3-5-sonnet-20240620
set model_name_short=claude
set lang=English
set model_class=Claude
set dtype=fp32
set generate_kwargs=config\generation_kwargs_causallm.json
set case_num=20
set diverse_plan=all
set output_dir=outputs\%model_name%\%lang%\resume
set test_per_case=1
set resume_epoch=14
set resume_input=0

python main_test.py --prompt_path %prompt_path% --questionnaire %questionnaire% --model_name %model_name% --model_name_short %model_name_short% --model_class %model_class% --dtype %dtype% --output_dir %output_dir% --generate_kwargs %generate_kwargs% --case_num %case_num% --diverse_plan %diverse_plan% --lang %lang% --test_per_case %test_per_case% --resume_epoch %resume_epoch% --resume_input %resume_input%