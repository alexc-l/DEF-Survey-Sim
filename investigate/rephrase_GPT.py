import os
import time

import openai
from tqdm import tqdm


def argument_setting():
    os.environ["http_proxy"] = "127.0.0.1:5878"
    os.environ["https_proxy"] = "127.0.0.1:5878"
    model = "gpt-3.5-turbo"
    openai.api_key = ""

    cn_scale_questions = [28, 62, 66, 67, 68, 69, 70, 72, 79, 101, 102, 103, 104, 105, 106, 113, 114, 115, 116, 117,
                          118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129]
    us_scale_questions = [28, 61, *range(65, 70), 71, 78, *range(100, 106), *range(112, 129), 160, *range(171, 178),
                          294]
    scale_questions = {
        "CN_survey": cn_scale_questions,
        "US_survey": us_scale_questions
    }
    cn_scale_questions_plus = [150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 161, 162]
    us_scale_questions_plus = [148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159]
    scale_questions_plus = {
        "CN_survey": cn_scale_questions_plus,
        "US_survey": us_scale_questions_plus
    }
    scaling = scale_questions["US_survey"] + scale_questions_plus["US_survey"]
    return model, scaling


def prepare_questions(scaling):
    with open("survey_data/extra/Chinese_questionaires.txt", "r", encoding="utf-8") as file:
        data = file.read()
        query_list = data.split("\n")

    question_list = []
    choice_list = []
    choice_number_list = []
    for i in range(len(query_list)):
        query = query_list[i]
        choice_number = query.split("] ")[0][1:]
        question = query.split("] ")[-1]
        if i + 1 not in scaling:
            question_list.append(question.split("- ", 1)[0])
            choice_list.append(question.split("- ", 1)[-1])
            choice_number_list.append(choice_number)
        else:
            question_list.append(question)
            choice_list.append("")
            choice_number_list.append(choice_number)

    return question_list, choice_list, choice_number_list


def rephrase(model, question_list, choice_list, choice_number_list, resume_count):
    input_count = 0
    with tqdm(total=len(question_list)) as pbar:
        for i in range(len(question_list)):
            question = question_list[i]
            if input_count != resume_count:
                input_count += 1
                pbar.update()
                continue
            messages = []
            message = {
                "role": "user",
                "content": f"重写下面的问题：{question}"
            }
            messages.append(message)

            start_time = time.time()
            try:
                response_dict = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=1,
                    top_p=1,
                )
            except:
                limit_time = time.time()
                time_gap = limit_time - start_time
                time_gap = time_gap * 1000 * 1000
                rest_time = time_gap % 60
                print(f'waiting response for {rest_time}')
                time.sleep(rest_time)
                start_time = time.time()
                response_dict = openai.ChatCompletion.create(
                    model=model,
                    messages=messages,
                    temperature=1,
                    top_p=1,
                )
                print(f"input_count: {i}")

            rephrased_question = response_dict['choices'][0]['message']['content']
            if choice_list[i] == "":
                rephrased_query = f"[{choice_number_list[i]}] {rephrased_question}"
            else:
                rephrased_query = f"[{choice_number_list[i]}] {rephrased_question}- {choice_list[i]}"
            with open("survey_data/extra/Chinese_rephrased_questionaires.txt", "a", encoding="utf-8") as file:
                file.write(f"{rephrased_query}\n")
            pbar.update()


def main():
    model, scaling = argument_setting()
    question_list, choice_list, choice_number_list = prepare_questions(scaling)
    rephrase(model, question_list, choice_list, choice_number_list, 93)


if __name__ == '__main__':
    main()
