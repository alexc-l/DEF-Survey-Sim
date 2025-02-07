import argparse
import json
import os
import time

import torch
# import openai
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import read_files

choices = {
    1: "指令理解能力：评估回复对于用户指令的完成度，首先查看回复中是否有以”Answer:“和”Explanantion:“开头的选择和解释，\n"
       "  其次查看选择部分中是否给出喜好倾向，或者能否捕捉到问题中的论述，是否完成功能指定任务（如提供观点、频率、数量等）。",
    2: "题型捕捉能力：评估回复的选择中是否符合问题的类型，即分辨单选、多选题、打分题等，并在多选题中能够选出多个选项作为回答，\n"
       "  同时不会超出选择的上限。",
    3: "无关表述输出: 评估回复内容是否完整，语句通顺，没有语法错误，并且没有和问题不相关的成分。",
    4: "角色扮演能力: 评估回复中时候能否真正做到模拟市民等角色扮演，如解释中谈到许多人，而没有使用我、我们等字眼，则视为没有具备该能力。",
    5: "表述价值冲突: 评估回复的价值倾向是否一致，即回答中的选择和解释的倾向是否匹配，主要关注选项和解释倾向是否相反的情况。",
    6: "提供错误事实: 评估回复陈述的事实是否属实，是否存在明显错误，对于一些宽泛的说法（如，”花费大量时间和精力“）可忽略。"
}


class Evaluator:

    def __init__(self, iqa_file_dict, qa_score, evaluated, fail_code, args):
        self.iqa_file_dict = iqa_file_dict
        self.qa_score = qa_score
        self.evaluated = evaluated
        self.args = args
        self.fail_code = fail_code
        # self.memory = []
        # self.post_pre = ""
        # self.sci_pre = ""
        # if self.args.lang == "English":
        #     self.multiple_choices_questions = multiple_choices_questions["US_survey"]
        #     self.scale_questions = scale_questions["US_survey"]
        #     self.scale_questions_plus = scale_questions_plus["US_survey"]
        #     self.factual_questions = factual_questions["US_survey"]
        #     self.post_index = utils.post_index["US_survey"]
        #     self.sci = utils.sci["US_survey"]
        # else:
        #     self.multiple_choices_questions = multiple_choices_questions["CN_survey"]
        #     self.scale_questions = scale_questions["CN_survey"]
        #     self.scale_questions_plus = scale_questions_plus["CN_survey"]
        #     self.factual_questions = factual_questions["CN_survey"]
        #     self.post_index = utils.post_index["CN_survey"]
        #     self.sci = utils.sci["CN_survey"]

    def intellect_eval_process(self):
        model_name = "/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/emnlp/cache/glm-4-9b-chat"
        # model_name = "/home/share/wuhkjdxyywyzstpu/home/ontoweb1/lhj/emnlp/llama-3-70b-Instruct-bnb-4bit"
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            # torch_dtype=torch.bfloat16,
            # low_cpu_mem_usage=True,
            trust_remote_code=True
        ).to("cuda").eval()

        with tqdm(total=len(self.iqa_file_dict), desc="Evaluating file:") as pbar:
            for file_path in self.iqa_file_dict:
                if file_path not in self.evaluated.keys():
                    self.evaluated[file_path] = []
                with tqdm(total=len(self.iqa_file_dict[file_path]), desc=file_path) as sbar:
                    for pair in self.iqa_file_dict[file_path]:
                        ques_num = pair[0]
                        # os.environ["http_proxy"] = "127.0.0.1:5878"
                        # os.environ["https_proxy"] = "127.0.0.1:5878"
                        # model = "gpt-3.5-turbo"
                        # # model = "gpt-3.5-turbo-1106"
                        # openai.api_key = ""

                        j_prompt = "You are a judge to determine whether a given response is appropriate for the corresponding question.\n"
                        j_aspect_dict = {
                            1: "You should evaluate the ability of Instruction comprehension. "
                               "You should assess the degree to which the response fulfills the user's instructions, "
                               "first by looking for choices and explanations that begin with 'Answer:' and 'Explantion:', "
                               "and second by looking for preferences given in the choices section. "
                               "Secondly, to see if preferences are given in the choice section, "
                               "or if the exposition in the question is captured, "
                               "and if the functionally specified tasks (e.g., providing opinions, frequency, quantity, etc.) are completed.\n",
                            2: "You should evaluate the ability of Question Capture Ability. "
                               "You should assess whether the responses match the type of question in their choices, "
                               "i.e., distinguishes between single choice, multiple choice, scaling questions, etc., "
                               "and is able to select more than one option as a response in a multiple choice question without exceeding the upper limit of choices.\n",
                            3: "You should evaluate the ability of Role-playing ability. "
                               "The ability to actually do role-playing, such as Sims, is assessed in the response, "
                               "and is not considered to be present if the explanation talks about many people without using words such as I and we.\n",
                            4: "You should evaluate the ability of Representation of value conflict: "
                               "You should assess the consistency of the value tendencies of the responses, "
                               "i.e., whether the choices and interpretive tendencies in the responses match, "
                               "focusing primarily on whether the choices and interpretive tendencies are reversed.\n",
                            5: "You should evaluate the ability of Provide Misstatements of Fact: "
                               "You should evaluate whether the facts stated in the response are true, whether there are obvious errors, "
                               "and ignore broad statements (e.g., 'a great deal of time and effort').\n",
                            6: "You should evaluate the ability of Irrelevant Expression Output: "
                               "You should evaluate whether the response is complete, well-spoken, free of grammatical errors, "
                               "and free of components that are not relevant to the question.\n"
                        }
                        j_post = "You sould give me a 'Yes' or 'No' answer, then present a analysis on the response."
                        for c_key in j_aspect_dict.keys():
                            j_input = f"{j_prompt}\n{j_aspect_dict[c_key]}\n{j_post}"
                            if file_path in self.evaluated.keys() and f"{ques_num}-{c_key}" not in self.evaluated[file_path]:

                                question = pair[2]
                                answer_exp_split = pair[4].split("Explanation:")
                                answer = answer_exp_split[0]
                                if len(answer_exp_split) > 1:
                                    exp = pair[4].split("Explanation:")[1]
                                    answer_exp = answer + " Explanation:" + exp
                                else:
                                    exp = ""
                                    answer_exp = answer

                                # message_sys = {
                                #     "role": "system",
                                #     "content": j_input
                                # }
                                req = f"Question: {question}\nResponse: {answer_exp}\nYour judge is:"
                                # message_usr = {
                                #     "role": "user",
                                #     "content": req
                                # }
                                # print('\n'+req+'\n')
                                # messages = [message_sys, message_usr]
                                response = GLM_4_api(model, tokenizer, j_input, req)
                                # start_time = time.time()
                                # try:
                                #     response_dict = openai.ChatCompletion.create(
                                #         model=model,
                                #         messages=messages,
                                #         temperature=1,
                                #         top_p=1,
                                #     )
                                # except:
                                #     limit_time = time.time()
                                #     time_gap = limit_time - start_time
                                #     time_gap = time_gap * 1000 * 1000
                                #     rest_time = time_gap % 60
                                #     if time_gap <= 10:
                                #         time_gap += 10
                                #     print(f'\nwaiting response for {rest_time}\n')
                                #     time.sleep(rest_time)
                                #     start_time = time.time()
                                #     response_dict = openai.ChatCompletion.create(
                                #         model=model,
                                #         messages=messages,
                                #         temperature=1,
                                #         top_p=1,
                                #     )
                                # response = response_dict['choices'][0]['message']['content']
                                # print(response)
                                juge_res = f"Judging {ques_num}: {response}"
                                out_dir = "gpt_judge"
                                with open(out_dir + os.sep + str(c_key) + "_judge.txt", "a", encoding='utf-8') as out_f:
                                    out_f.write(juge_res + '\n')
                                if 'No' in response:
                                    self.qa_score[str(c_key)][ques_num] += 1
                                elif 'Yes' in response:
                                    self.qa_score[str(self.fail_code)][ques_num] += 1
                                self.evaluated[file_path].append(f"{ques_num}-{c_key}")

                                self.qa_score.to_csv(self.args.data_storage, index=True)
                                with open(out_dir + f"/{args.lang}-evaluated.json", "w", encoding="utf-8") as hf:
                                    json.dump(self.evaluated, hf)
                            else:
                                continue
                        sbar.update()
                pbar.update()


def GLM_4_api(model, tokenizer, sys_msg, request):
    inputs = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": sys_msg},
            {"role": "user", "content": request}
        ],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    ).to("cuda")

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]

    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def argument_parser():
    parser = argparse.ArgumentParser(description='LLM Value Investigation args', epilog='Information end')
    parser.add_argument(
        "--lang",
        type=str,
        default="Chinese",
        help="questionnaire language")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/Chinese_chatglm",
        help="model response dir"
    )
    parser.add_argument(
        "--data_storage",
        type=str,
        default="human_eval/qa_scores.csv",
        help="data storage path, direct to a excel file"
    )
    parser.add_argument(
        "--questionnaire",
        type=str,
        default="survey_data/sample/",
        help="Questionnaire dir path")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="survey_data/introse/CN_intellect_instructions.txt",
        help="instruction file path")
    parser.add_argument(
        "--intellect",
        type=str,
        default="yes",
        help="whether do intellect measure"
    )
    return parser


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    if not os.path.exists(args.data_storage):
        os.makedirs(args.data_storage)
    args.data_storage += "/AECE_score.csv"
    iqa_file_dict, qa_score, evaluated, fail_code = read_files(args=args)
    evaluator = Evaluator(
        iqa_file_dict=iqa_file_dict, qa_score=qa_score, evaluated=evaluated, fail_code=fail_code, args=args
    )
    evaluator.intellect_eval_process()
