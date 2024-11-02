set lang="zh_English"
set output_dir="D:\projects\LLMValueInvestigation\code\investigate\outputs\lang_abla\zh_English\dolphin-2.2.1-mistral-7b"
set data_storage="D:\projects\LLMValueInvestigation\code\investigate\outputs\lang_abla\zh_English\dolphin-2.2.1-mistral-7b\qa_score.csv"
set questionnaire="D:\projects\LLMValueInvestigation\code\investigate\survey_data\AECE\\"

python auto_analyze.py --lang %lang% --output_dir %output_dir% --data_storage %data_storage% --questionnaire %questionnaire%