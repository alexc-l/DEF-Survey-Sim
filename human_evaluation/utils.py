import json
import re
import os

import pandas as pd

lang_to_country = {"English": ["United States"], "Chinese": ["中国"], "en_Chinese": ["China"], "zh_English": ["美国"]}
US_states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
             "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
             "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
             "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
             "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
             "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
CH_states = ["河北省", "山西省", "辽宁省", "吉林省", "黑龙江省", "江苏省", "浙江省", "安徽省", "福建省", "江西省",
             "山东省", "河南省",
             "湖北省", "湖南省", "广东省", "海南省", "四川省", "贵州省", "云南省", "陕西省", "甘肃省", "青海省",
             "台湾省"]
country_to_states = {"United States": US_states, "China": CH_states}

cn_multiple_choices_questions = [7, 8, *range(210, 214), 227, 228, *range(430, 434), 447, 471, 513]
# cn_multiple_choices_questions = [7, 31, 73]
us_multiple_choices_questions = [7, 8, 304, 305]
multiple_choices_questions = {
    "CN_survey": cn_multiple_choices_questions,
    "US_survey": us_multiple_choices_questions
}

cn_scale_questions = [28, 62, 66, 67, 68, 69, 70, 72, 79, 101, 102, 103, 104, 105, 106, 113, 114, 115, 116, 117, 118,
                      119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 248, 282, 286, 287, 288, 289, 290, 292,
                      299, 321, 322, 323, 324, 325, 326, 333, 334, 335, 336, 337, 338, 339, 340, 341, 342, 343, 344,
                      345, 346, 347, 348, 349, 449, 451, 464, 466, 475, 482, 490, 491, 496, 500, 508, 521, 524, ]


us_scale_questions = [28, 61, 65, 66, 67, 68, 69, 71, 78, 100, 101, 102, 103, 104, 105, 112, 113, 114, 115, 116, 117,
                      118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 160, 171, 172, 173, 174, 175, 176, 177,
                      294, 325, 358, 362, 363, 364, 365, 366, 368, 375, 397, 398, 399, 400, 401, 402, 409, 410, 411,
                      412, 413, 414, 415, 416, 417, 418, 419, 420, 421, 422, 423, 424, 425, 457, 468, 469, 470, 471,
                      472, 473, 474, 591]

scale_questions = {
    "CN_survey": cn_scale_questions,
    "US_survey": us_scale_questions
}
cn_scale_questions_plus = [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162, 370, 371, 372, 373, 374,
                           375, 376, 377, 378, 379, 380, 381, 382, 480, 509, 516, 525, 530, 539]

us_scale_questions_plus = [148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 445, 446, 447, 448, 449, 450,
                           451, 452, 453, 454, 455, 456]

scale_questions_plus = {
    "CN_survey": cn_scale_questions_plus,
    "US_survey": us_scale_questions_plus
}

cn_factual_questions = [63, 64, 65, 282, 283, 284]

us_factual_questions = [62, 63, 64, 359, 360, 361]
factual_questions = {
    "CN_survey": cn_factual_questions,
    "US_survey": us_factual_questions
}

us_post_index = [(94, 96, 98, 391, 393, 395), (95, 97, 99, 392, 394, 396)]
cn_post_index = [(95, 97, 99, 315, 317, 319), (96, 98, 100, 316, 318, 320)]
post_index = {
    "CN_survey": cn_post_index,
    "US_survey": us_post_index
}

us_sci = [(100, 397), (101, 398)]
cn_sci = [(101, 321), (106, 326)]
sci = {
    "CN_survey": cn_sci,
    "US_survey": us_sci
}

self_aware_keywords = "As an artificial intelligence assistant|As an AI language model|As a language model|作为一个人工智能助手|作为一个大语言模型"


def get_instructions(instruction_path):
    with open(instruction_path, "r",
              encoding="utf-8") as instruct_file:
        instruct_ = instruct_file.read()
        instructs = instruct_.split("\n")

    qi_dict = {}
    for instruct in instructs:
        if instruct != "\n":
            question_nums = instruct.split("]")[0][1:]
            instruction = instruct.split("]")[1]
            if "-" in question_nums:
                question_ns = question_nums.split("-")
                for n in range(int(question_ns[0]), int(question_ns[1]) + 1):
                    qi_dict[n] = instruction
            else:
                question_ns = int(question_nums)
                qi_dict[question_ns] = instruction
    return qi_dict


def read_files(args):
    # Cipher logs
    file_log_dict = {}
    file_log_dict[args.output_dir] = sorted(get_files(args.output_dir))
    # Cipher questions
    qa_list, choice_num_list, question_list = get_choice_num_list(lang=args.lang, questionnaire_path=args.questionnaire)

    # Cipher instructions
    qi_dict = get_instructions(args.instruction_path)

    # Cipher answers
    iqa_file_dict = {}
    for file in file_log_dict[args.output_dir]:
        log_file = file.split(".txt")[0] + ".log"
        with open(log_file, 'r', encoding="utf-8") as log:
            random_index = log.read().split('\n')
        # print(random_index)
        # print(file)
        iqa_pairs = []
        answers = prepare_answers(choices_list=choice_num_list, output_file=file)
        for ques_num in range(len(answers)):
            if str(ques_num) in random_index:
                iqa_pairs.append((ques_num, qi_dict[ques_num], question_list[ques_num], qa_list[ques_num],
                                  answers[ques_num]))

        iqa_file_dict[file] = iqa_pairs

    if os.path.exists(args.data_storage):
        qa_score = pd.read_csv(args.data_storage, index_col=0)
    else:
        columns = range(max(choice_num_list) + 2)
        columns = list(map(str, columns))
        qa_score = pd.DataFrame(columns=columns, index=range(len(choice_num_list) + 1), data=0)

    out_dir = "gpt_judge"
    if os.path.exists(out_dir + f"/{args.lang}-evaluated.json"):
        with open(out_dir + f"/{args.lang}-evaluated.json", "r", encoding="utf-8") as history_file:
            evaluated = json.load(history_file)
    else:
        evaluated = {}
    return iqa_file_dict, qa_score, evaluated, max(choice_num_list) + 1


def prepare_answers(output_file, choices_list):
    with open(output_file, "r",
              encoding="utf-8") as ans:
        answers_raw = ans.read()
        answers_raw = answers_raw.split("\n")

    answers = []
    print(output_file)
    for i in range(len(answers_raw)):
        pattern = re.compile(r'Q\d+:', re.I)
        if re.match(pattern, answers_raw[i]):
            if "base" in output_file:
                ans_split = answers_raw[i].split(": ", 1)
                answers_raw[i] = ans_split[0] + ": Answer: " + ans_split[1]
            answer = answers_raw[i] + " "
            answers.append(answer)
        elif answers_raw != "\n":
            answers[-1] += answers_raw[i] + " "

    assert len(answers) == len(choices_list), "length mismatch: ans %d, ques %d" % (len(answers), len(choices_list))

    return answers


def get_choice_num_list(lang, questionnaire_path):
    questionaire_path = questionnaire_path + lang + "_questionaires.txt"
    with open(questionaire_path, "r", encoding="utf-8") as q:
        ques = q.read()
        questions = ques.split("\n")
    choice_num_list = []
    qa_list = []
    question_list = []

    for q in questions:
        if q != "\n":
            q = q.replace("[COUNTRY]", lang_to_country[lang][0])
            choice_split = q.split("]")
            _choices = choice_split[0]
            choices_text = choice_split[1].split("- ", 1)  # scale
            choices_text_dict = {}
            if len(choices_text) > 1:
                choices_text_list = re.split(':|\. ', choices_text[1])
                for i in range(0, len(choices_text_list), 2):
                    # print(choices_text_list)
                    choice_text = choices_text_list[i + 1].strip()
                    choice_num = int(choices_text_list[i])
                    choices_text_dict[choice_num] = choice_text
            choices = _choices.split("[")[1]
            qa_list.append(choices_text_dict)
            choice_num_list.append(int(choices))

            question = q.split("]")[1]
            question_list.append(question)
    return qa_list, choice_num_list, question_list


def get_files(output_dir):
    file_paths = []
    for filepath, dirnames, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.endswith("responses.txt"):
                file_paths.append(os.path.join(filepath, filename))
    return file_paths
