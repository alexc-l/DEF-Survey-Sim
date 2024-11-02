SETLOCAL ENABLEDELAYEDEXPANSION
set lang=English
set output_dir=D:\projects\LLMValueInvestigation\code\human_evaluation\outputs\US_survey_all\baichuan2-13B-4bits\
set data_storage=D:\projects\LLMValueInvestigation\code\human_evaluation\outputs\US_survey_all\baichuan2-13B-4bits\
set questionnaire=D:\projects\LLMValueInvestigation\code\human_evaluation\survey_data\AECE_v1\
for %%i in (0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19) do (
    set out_dir=%output_dir%%%i
    set d_storage=%data_storage%%%i\qa_score_single.csv
python auto_analyze.py --lang %lang% --output_dir !out_dir! --data_storage !d_storage! --questionnaire %questionnaire%)