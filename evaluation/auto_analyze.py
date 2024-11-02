import argparse
import logging
import os
import re

import pandas as pd
from tqdm.auto import tqdm

cn_multiple_choices_questions = [7, 8, *range(210, 214)]
us_multiple_choices_questions = [7, 8]
multiple_choices_question = {
    "CN_survey": cn_multiple_choices_questions,
    "US_survey": us_multiple_choices_questions
}


def get_choice_num_list(lang, questionnaire_path):
    questionaire_path = questionnaire_path + lang + "_questionaires.txt"
    with open(questionaire_path, "r", encoding="utf-8") as q:
        ques = q.read()
        questions = ques.split("\n")
    choice_num_list = []
    qa_list = []
    for q in questions:
        if q != "\n":
            choice_split = q.split("]")
            _choices = choice_split[0]
            choices_text = choice_split[1].split("- ", 1)  # scale
            choices_text_dict = {}
            if len(choices_text) > 1:
                choices_text_list = re.split(':|\.', choices_text[1])
                print(choices_text_list)
                for i in range(0, len(choices_text_list), 2):
                    # assert choices_text_list[i].strip().isdigit(), f"not a choice number {choices_text_list[i]}"
                    choice_text = choices_text_list[i + 1].strip()
                    # print(choices_text_list)
                    # print(choices_text)
                    choice_num = int(choices_text_list[i])
                    choices_text_dict[choice_num] = choice_text
            choices = _choices.split("[")[1]
            qa_list.append(choices_text_dict)
            choice_num_list.append(int(choices))
    return qa_list, choice_num_list


def prepare_answers(output_file, choices_list):
    print(output_file)
    with open(output_file, "r",
              encoding="utf-8") as ans:
        answers_raw = ans.read()
        answers_raw = answers_raw.split("\n")

    answers = []

    for i in range(len(answers_raw)):
        pattern = re.compile(r'Q\d+', re.I)
        if re.match(pattern, answers_raw[i]):
            if "base" in output_file:
                ans_split = answers_raw[i].split(": ", 1)
                answers_raw[i] = ans_split[0] + ": Answer: " + ans_split[1]
            answer = answers_raw[i]
            answers.append(answer)
        elif answers_raw != "\n":
            answers[-1] += answers_raw[i]

    # assert len(answers) == len(choices_list), "length mismatch: ans %d, ques %d" % (len(answers), len(choices_list))

    return answers


def process_data_for_one_epoch(lang, choices_list, answers, out_file, qa_list):
    if lang == "English":
        multiple_choices_questions = multiple_choices_question["US_survey"]
    else:
        multiple_choices_questions = multiple_choices_question["CN_survey"]

    qa_pairs = {}
    if os.path.exists(out_file):
        qa_score = pd.read_csv(out_file, index_col=0, )
    else:
        columns = range(max(choices_list) + 2)
        columns = list(map(str, columns))
        qa_score = pd.DataFrame(columns=columns, index=range(len(choices_list) + 1), data=0)

    for answer in answers:
        question_num = [int(s) for s in re.findall(r'-?\d+?\d*', answer.split(":", 1)[0])][0]
        answer_body = answer.split("Answer:", 1)[-1]
        answer_end = answer_body.find("Explanation:")
        if answer_end != -1:
            answer_body = answer_body[:answer_end]
        choices = [int(s) for s in re.findall(r'-?\d+?\d*', answer_body)]
        qa_pairs[question_num] = choices
        found_choice = []
        pattern = re.compile(r'some .* others', re.I)
        if len(re.findall(pattern, answer_body)) > 0:
            choices = [-1]
            print(f"Fail to complete the task detected: {re.findall(pattern, answer_body)}")
        elif len(qa_pairs[question_num]) > 1:
            if "base" in out_file:
                if len(answer) != len(set(answer)) or len(set(answer)) <= 5:
                    found_choice.append(12)
                    break
            if question_num not in multiple_choices_questions:
                if answer_body.lower().find("scale of 1 to 10") != -1:
                    choices.remove(1)
                    choices.remove(10)
                    found_choice = choices
                for key in qa_list[question_num - 1]:
                    if answer_body.lower().find(qa_list[question_num - 1][key].lower()) != -1:
                        found_choice.append(key)
                        break
                if len(found_choice) == 0:
                    logging.warning(f'warning, quenstion {question_num} have multiple answers {choices}, please check')
                    continue
        elif len(qa_pairs[question_num]) == 0:
            for key in qa_list[question_num - 1]:
                if answer_body.lower().find(qa_list[question_num - 1][key].lower()) != -1:
                    found_choice.append(key)
                    break
            if len(found_choice) == 0:
                logging.warning(f"warning, quenstion {question_num} have no answers {choices}, please check.")
                continue

        if len(found_choice) != 0:
            choices = found_choice
        if question_num == 7:
            if len(choices) > 5:
                choices = choices[:5]
            else:
                choices = choices
        for choice in choices:
            if choice >= 0 and choice <= choices_list[question_num - 1]:
                qa_score[str(choice)][question_num] += 1
            elif choice == -1:
                qa_score[str(max(choices_list) + 1)][question_num] += 1
            else:
                logging.warning(f"warning, quenstion {question_num} have excedding choice {choices}, please check.")

    qa_score.to_csv(out_file, index=True)


def get_files(output_dir):
    file_paths = []
    for filepath, dirnames, filenames in os.walk(output_dir):
        for filename in filenames:
            if filename.endswith("responses.txt"):
                file_paths.append(os.path.join(filepath, filename))
    return file_paths


def argument_parser():
    parser = argparse.ArgumentParser(description='LLM Value Investigation args', epilog='Information end')
    parser.add_argument(
        "--lang",
        type=str,
        required=True,
        help="questionnaire language")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="model response dir"
    )
    parser.add_argument(
        "--data_storage",
        type=str,
        required=True,
        help="data storage path, direct to a excel file"
    )
    parser.add_argument(
        "--questionnaire",
        type=str,
        required=True,
        help="Questionnaire dir path")
    return parser


def main():
    parser = argument_parser()
    args = parser.parse_args()
    logging.basicConfig(level=logging.DEBUG,
                        filename=args.output_dir + "/data_processing.log",
                        format="%(asctime)s - %(name)s - %(levelname)-9s - %(filename)-8s : %(lineno)s line - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    qa_dict_list, choice_num_list = get_choice_num_list(lang=args.lang, questionnaire_path=args.questionnaire)
    qa_list = []
    for k_v in qa_dict_list:
        temp = sorted(k_v.items(), key=lambda v: len(v[1]), reverse=True)
        choice_dict = {}
        for group in temp:
            choice_dict[group[0]] = group[1]
        qa_list.append(choice_dict)

    file_paths = get_files(output_dir=args.output_dir)
    with tqdm(total=len(file_paths))as pbar:
        for i in range(len(file_paths)):
            if i > 0:
                with open(file_paths[i], "r",
                          encoding="utf-8") as ansp:
                    answers_raw_prev = ansp.read()
                with open(file_paths[i - 1], "r",
                          encoding="utf-8") as ans:
                    answers_raw = ans.read()
                if answers_raw_prev == answers_raw:
                    logging.info(f"The two responses are the same: {file_paths[i - 1]} and {file_paths[1]}")
                    pbar.update()
                    continue
            answers = prepare_answers(choices_list=choice_num_list, output_file=file_paths[i])
            # print(a)
            process_data_for_one_epoch(lang=args.lang, choices_list=choice_num_list, answers=answers, out_file=args.data_storage,
                                       qa_list=qa_list)
            logging.info(f"data analysis for {file_paths[i]} is complete ...")
            pbar.update()


if __name__ == '__main__':
    main()
