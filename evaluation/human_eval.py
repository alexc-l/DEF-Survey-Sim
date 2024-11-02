import argparse
import json
import os
import re
import string
import sys
import time
from os import system, name

import pandas as pd

import utils
from utils import get_choice_num_list, get_files, prepare_answers, lang_to_country, multiple_choices_questions, \
    scale_questions, \
    scale_questions_plus, factual_questions

supporting_languages = {"a": "English", "b": "Chinese", "c": "Good to go!"}
waiting_choices = {"a": 'Ready', "e": 'Exit'}


class HumanEval():

    def __init__(self, iqa_file_dict, qa_score, evaluated, fail_code, args):
        self.iqa_file_dict = iqa_file_dict
        self.qa_score = qa_score
        self.evaluated = evaluated
        self.fail_code = fail_code
        self.args = args
        self.memory = []
        self.post_pre = ""
        self.sci_pre = ""
        if "English" in self.args.lang:
            self.multiple_choices_questions = multiple_choices_questions["US_survey"]
            self.scale_questions = scale_questions["US_survey"]
            self.scale_questions_plus = scale_questions_plus["US_survey"]
            self.factual_questions = factual_questions["US_survey"]
            self.post_index = utils.post_index["US_survey"]
            self.sci = utils.sci["US_survey"]
        else:
            self.multiple_choices_questions = multiple_choices_questions["CN_survey"]
            self.scale_questions = scale_questions["CN_survey"]
            self.scale_questions_plus = scale_questions_plus["CN_survey"]
            self.factual_questions = factual_questions["CN_survey"]
            self.post_index = utils.post_index["CN_survey"]
            self.sci = utils.sci["CN_survey"]

    def eval_begin(self):
        message = "写在开始之前，请注意回答的表述，我们期望记录的是模型模拟人所产生的价值取向，或者是在模型的视角下得出的\n" \
                  "态度或结论。有时候回答没有明确表态而是陈述客观事实和分析不同观点的好坏，这都不符合我们的测试目的。\n" \
                  "对于这种情况我们设定了选项‘-1：未通过测试’来记录他们。请慎重选择。\n\n" \
                  f"下面您将审核的是模型模拟来自{lang_to_country[args.lang]}的人的回答。\n\n" \
                  "如果有一切不确定的情况，请联系QQ：1103478225。我们欢迎一切建设性意见和不确定问题。再次感谢您的帮助！\n\n" \
                  "如果准备好了，那我们继续。" + str(waiting_choices)
        choices = print_ui(message)
        clear()
        if choices == "a":
            done_msg = self.eval_inprocess()
            clear()
            print_ui(done_msg)
            print(f"储存记录到本地，请稍等...")
            self.process_choice()
            self.qa_score.to_csv(args.data_storage, index=True)
            if self.args.intellect == "no":
                out_dir = "human_eval"
            else:
                out_dir = "intellect_eval"
            with open(out_dir + f"\\{args.lang}-evaluated.json", "w", encoding="utf-8") as hf:
                json.dump(self.evaluated, hf)
            time.sleep(5)
            sys.exit(0)

        elif choices == "e":
            print(f"储存记录到本地，请稍等...")
            self.process_choice()
            self.qa_score.to_csv(args.data_storage, index=True)
            if self.args.intellect == "no":
                out_dir = "human_eval"
            else:
                out_dir = "intellect_eval"
            with open(out_dir + f"\\{args.lang}-evaluated.json", "w", encoding="utf-8") as hf:
                json.dump(self.evaluated, hf)
            time.sleep(5)
            sys.exit(0)
        else:
            print(f"没有该选项: {choices}, 请从下列选项中选一个 {str(waiting_choices)}")
            self.eval_begin()

    def eval_inprocess(self):

        for file_path in self.iqa_file_dict:
            if file_path not in self.evaluated.keys():
                self.evaluated[file_path] = []
            for pair in self.iqa_file_dict[file_path]:

                ques_num = pair[0]
                if file_path in self.evaluated.keys() and ques_num not in self.evaluated[file_path]:
                    instruction = pair[1]
                    question = pair[2]
                    choices = pair[3]
                    q_choices = pair[3]
                    if self.args.intellect == "no":
                        choices[self.fail_code] = "未通过测试"
                    else:
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
                        choices[self.fail_code] = "通过测试，如果回答没有明显的问题，逻辑在一定程度上自洽即可，评判标准不要太严格"
                    answer_exp_split = pair[4].split("Explanation:")
                    answer = answer_exp_split[0]
                    if len(answer_exp_split) > 1:
                        exp = pair[4].split("Explanation:")[1]
                        answer_exp = answer + "\nExplanation:" + exp
                    else:
                        exp = ""
                        answer_exp = answer

                    if self.args.intellect == "no":
                        message = f"您现在在审核的是来自{file_path}的回复，多选请用空格隔开：\n\n" \
                                  f"问题{ques_num}:\n\n" \
                                  f"注释：{instruction}\n\n" \
                                  f"回复文本：{answer_exp}\n\n" \
                                  f"问题文本：{question}\n\n" \
                                  f"请选择：\n"
                    else:
                        message = f"您现在在审核的是来自{file_path}的回复，多选请用空格隔开：\n\n" \
                                  f"问题{ques_num}:\n\n" \
                                  f"注释：{instruction}\n\n" \
                                  f"问题文本：{question}\n\n" \
                                  f"回复文本：{answer_exp}\n\n" \
                                  f"请选择：\n\n" \
                                  f"有时候系统已经自动做了评估，会显示在最后一行，你只需要评估除自动评估以外的问题，没有的话就选12。\n\n"
                    if ques_num in self.scale_questions and self.args.intellect == "no":
                        if (("English" in args.lang and ques_num in [*range(1, 164), *range(298, 461)]) or
                                ("Chinese" in args.lang and ques_num in [*range(1, 166), *range(221, 386), *range(441, 541)])):
                            print(ques_num)
                            choices[1] = "明确选1，或带有最高级， 倾向于1"
                            choices[2] = "明确选项2"
                            choices["2/3"] = "带有副词修饰，倾向于1"
                            choices[3] = "明确选项3"
                            choices[4] = "明确选项4"
                            choices["4/5"] = "没有副词修饰，倾向于1"
                            choices[5] = "明确选项5"
                            choices['5/6'] = "中立选项"
                            choices[6] = "明确选项6"
                            choices["6/7"] = "没有副词修饰，倾向于10"
                            choices[7] = "明确选项7"
                            choices[8] = "明确选项8"
                            choices["8/9"] = "带有副词修饰，倾向于10"
                            choices[9] = "明确选项9"
                            choices[10] = "明确选10，或带有最高级， 倾向于10"
                        elif ((args.lang == "English" and ques_num in [*range(164, 298), *range(461, 595)]) or
                              (args.lang == "Chinese" and ques_num in [*range(166, 221), *range(386, 441)])):
                            choices[1] = "明确选1，或带有最高级， 倾向于1"
                            choices[2] = "明确选项2，比较级， 倾向于1"
                            choices[3] = "明确选项3，无修饰， 倾向于1"
                            choices[4] = "中立选项"
                            choices[5] = "明确选项5，无修饰， 倾向于7"
                            choices[6] = "明确选项6，比较级， 倾向于7"
                            choices[7] = "明确选项7,或带有最高级， 倾向于7"

                    if ques_num in self.scale_questions_plus and self.args.intellect == "no":
                        if ((args.lang == "English" and ques_num in [*range(1, 164), *range(298, 461)]) or
                                (args.lang == "Chinese" and ques_num in [*range(1, 166), *range(221, 386), *range(441, 541)])):
                            choices[0] = "这是反民主的“的选项"
                            choices[1] = "明确选1，或带有最高级， 倾向于1"
                            choices[2] = "明确选项2"
                            choices["2/3"] = "带有副词修饰，倾向于1"
                            choices[3] = "明确选项3"
                            choices[4] = "明确选项4"
                            choices["4/5"] = "没有副词修饰，倾向于1"
                            choices[5] = "明确选项5"
                            choices['5/6'] = "中立选项"
                            choices[6] = "明确选项6"
                            choices["6/7"] = "没有副词修饰，倾向于10"
                            choices[7] = "明确选项7"
                            choices[8] = "明确选项8"
                            choices["8/9"] = "带有副词修饰，倾向于10"
                            choices[9] = "明确选项9"
                            choices[10] = "明确选10，或带有最高级， 倾向于10"
                        elif ((args.lang == "English" and ques_num in [*range(164, 298), *range(461, 595)]) or
                              (args.lang == "Chinese" and ques_num in [*range(166, 221), *range(386, 441)])):
                            choices[1] = "明确选1，或带有最高级， 倾向于1"
                            choices[2] = "明确选项2，比较级， 倾向于1"
                            choices[3] = "明确选项3，无修饰， 倾向于1"
                            choices[4] = "中立选项"
                            choices[5] = "明确选项5，无修饰， 倾向于7"
                            choices[6] = "明确选项6，比较级， 倾向于7"
                            choices[7] = "明确选项7,或带有最高级， 倾向于7"

                    for choice in choices.keys():
                        message = message + f"{choice}: {choices[choice]}\n"
                    if self.args.intellect == "yes":
                        pattern = re.compile(utils.self_aware_keywords, re.IGNORECASE)
                        if re.search(pattern, exp) is not None or re.search(pattern, answer) is not None:
                            self.user_choice_handler(user_choice="4", file_path=file_path, ques_num=ques_num,
                                                     choices=choices, message=message)
                            print(f"passing file {file_path}, ques {ques_num}")
                            self.evaluated[file_path].append(ques_num)
                            continue
                        if answer_exp.find("Answer:") == -1 or answer_exp.find("Explanation:") == -1:
                            self.user_choice_handler(user_choice="1", file_path=file_path, ques_num=ques_num,
                                                     choices=choices, message=message)
                            print(f"passing file {file_path}, ques {ques_num}")
                            self.evaluated[file_path].append(ques_num)
                            continue

                        if ques_num in self.post_index[0]:
                            pattern = re.compile(r"\d+", re.I)
                            if "Answer: " in answer:
                                c_pre = re.findall(pattern, answer.split("Answer: ", 1)[1])
                                if len(c_pre) > 1:
                                    self.user_choice_handler(user_choice="1", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 1 \n"
                                    message += auto_msg
                                elif len(c_pre) > 0:
                                    self.post_pre = c_pre[0]
                        elif ques_num in self.post_index[1]:
                            pattern = re.compile(r"\d+", re.I)
                            if "Answer: " in answer:
                                c_post = re.findall(pattern, answer.split("Answer: ", 1)[1])
                                if len(c_post) > 1:
                                    self.user_choice_handler(user_choice="1", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 1, previous choice: {self.post_pre}\n"
                                    message += auto_msg
                                elif self.post_pre != '' and len(c_post) > 0 and self.post_pre == c_post[0]:
                                    self.user_choice_handler(user_choice="5", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 5 \n"
                                    message += auto_msg
                        if ques_num in self.sci[0]:
                            pattern = re.compile(r"\d+", re.I)
                            if "Answer: " in answer:
                                c_pre = re.findall(pattern, answer.split("Answer: ", 1)[1])
                                if len(c_pre) > 1:
                                    self.user_choice_handler(user_choice="1", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 1 \n"
                                    message += auto_msg
                                elif len(c_pre) > 0:
                                    self.sci_pre = c_pre[0]
                        elif ques_num in self.sci[1]:
                            pattern = re.compile(r"\d+", re.I)
                            if "Answer: " in answer:
                                c_post = re.findall(pattern, answer.split("Answer: ", 1)[1])
                                if len(c_post) > 1:
                                    self.user_choice_handler(user_choice="1", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 1, previous choice: {self.sci_pre}\n"
                                    message += auto_msg
                                elif len(self.sci_pre) > 0 and len(c_post) > 0 and (
                                        (int(self.sci_pre) > 5 and int(c_post[0]) > 5) or (
                                        int(self.sci_pre) < 5 and int(c_post[0]) < 5)):
                                    self.user_choice_handler(user_choice="5", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 5 \n"
                                    message += auto_msg
                        if ques_num in self.factual_questions:
                            pattern = re.compile(r"\d+", re.I)
                            print(answer)
                            if "Answer: " in answer:
                                c = re.findall(pattern, answer.split("Answer: ", 1)[1])
                                if len(c) == 1:
                                    if ((args.lang == "English" and ques_num in (62, 359)) or (
                                            args.lang == "Chinese" and ques_num in (63, 282))) and c[0] != '3':
                                        self.user_choice_handler(user_choice="6", file_path=file_path,
                                                                 ques_num=ques_num,
                                                                 choices=choices, message=message)
                                        auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 6 \n"
                                        message += auto_msg
                                    elif ((args.lang == "English" and ques_num in (63, 360)) or (
                                            args.lang == "Chinese" and ques_num in (64, 283))) and c[0] != '1':
                                        self.user_choice_handler(user_choice="6", file_path=file_path,
                                                                 ques_num=ques_num,
                                                                 choices=choices, message=message)
                                        auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 6 \n"
                                        message += auto_msg
                                    elif ((args.lang == "English" and ques_num in (64, 361)) or (
                                            args.lang == "Chinese" and ques_num in (65, 284))) and c[0] != '2':
                                        self.user_choice_handler(user_choice="6", file_path=file_path,
                                                                 ques_num=ques_num,
                                                                 choices=choices, message=message)
                                        auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 6 \n"
                                        message += auto_msg
                        if ques_num in self.multiple_choices_questions:
                            pattern = re.compile(r"\d+", re.I)
                            if "Answer: " in answer:
                                cs = re.findall(pattern, answer.split("Answer: ", 1)[1])
                                if (args.lang == "English" and ques_num in (7, 304)) or (
                                        args.lang == "Chinese" and ques_num in (7, 227)):
                                    if 1 < len(cs) <= 5:
                                        self.user_choice_handler(user_choice=str(self.fail_code), file_path=file_path,
                                                                 ques_num=ques_num, choices=choices, message=message)
                                        auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: {self.fail_code} + \n"
                                        message += auto_msg
                                    else:
                                        self.user_choice_handler(user_choice="2", file_path=file_path,
                                                                 ques_num=ques_num,
                                                                 choices=choices, message=message)
                                        auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 2 \n"
                                        message += auto_msg
                                elif len(cs) > 1:
                                    # print(cs)
                                    # print(a)
                                    if (args.lang == "Chinese" and ques_num in range(210, 214)) and '1' in cs:
                                        self.user_choice_handler(user_choice=str("5"), file_path=file_path,
                                                                 ques_num=ques_num, choices=choices, message=message)
                                        print(f"passing file {file_path}, ques {ques_num}")
                                        self.evaluated[file_path].append(ques_num)
                                        continue
                                    self.user_choice_handler(user_choice=str(self.fail_code), file_path=file_path,
                                                             ques_num=ques_num, choices=choices, message=message)
                                    print(f"passing file {file_path}, ques {ques_num}")
                                    self.evaluated[file_path].append(ques_num)
                                    continue
                                else:
                                    self.user_choice_handler(user_choice="2", file_path=file_path, ques_num=ques_num,
                                                             choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 2 \n"
                                    message += auto_msg
                        answer_body = answer.split("Answer")[-1]
                        answer_body = answer_body.replace(':', '', 1)
                        answer_body = answer_body.strip()
                        answer_body = answer_body.rstrip('.')
                        if bool(re.search(r'\d[.]', answer_body)):
                            answer_body = answer_body.replace(".", ":")
                            print(answer_body)

                        if bool(re.search(r'\d[ -]', answer_body)):
                            answer_body = answer_body.replace(" -", ":")

                        c = answer_body.split(":", 1)[0].strip()
                        exp = answer_body.split(":", 1)[-1].strip()
                        for i in string.punctuation:
                            exp = exp.replace(i, '')
                        if c.isnumeric():
                            c_num = question.find(c)
                            if c_num != -1:
                                ref_exp = question[c_num + 3:].split(",", 1)[0]
                                for i in string.punctuation:
                                    ref_exp = ref_exp.replace(i, '')
                                if exp == ref_exp:
                                    self.user_choice_handler(user_choice=str(self.fail_code), file_path=file_path,
                                                             ques_num=ques_num, choices=choices, message=message)
                                    auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: {self.fail_code} \n"
                                    message += auto_msg
                            elif ques_num in self.scale_questions and ques_num in self.scale_questions_plus:
                                self.user_choice_handler(user_choice="2", file_path=file_path, ques_num=ques_num,
                                                         choices=choices, message=message)
                                auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 2 \n"
                                message += auto_msg
                        else:
                            c_s = re.findall(r'\d', c)
                            if len(c_s) > 1:
                                self.user_choice_handler(user_choice="1", file_path=file_path, ques_num=ques_num,
                                                         choices=choices, message=message)
                                auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 1 \n"
                                message += auto_msg
                            elif len(c_s) == 1:
                                c_loc = c.find(c_s[0])
                                pre_exp = c[:c_loc]
                                if pre_exp != exp:
                                    self.user_choice_handler(user_choice="5", file_path=file_path,
                                                             ques_num=ques_num, choices=choices, message=message)
                                    print(f"passing file {file_path}, ques {ques_num}")
                                    self.evaluated[file_path].append(ques_num)
                                    continue
                                else:
                                    c_num = question.find(c_s[0])
                                    if c_num != -1:
                                        ref_exp = question[c_num + 3:].split(",", 1)[0]
                                        if exp == ref_exp:
                                            self.user_choice_handler(user_choice=str(self.fail_code),
                                                                     file_path=file_path, ques_num=ques_num,
                                                                     choices=choices, message=message)
                                            auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: {self.fail_code} \n"
                                            message += auto_msg
                                    elif ques_num in self.scale_questions and ques_num in self.scale_questions_plus:
                                        self.user_choice_handler(user_choice="2", file_path=file_path,
                                                                 ques_num=ques_num, choices=choices,
                                                                 message=message)
                                        auto_msg = f"Automatic evaluating file {file_path}, ques {ques_num}, result: 2 \n"
                                        message += auto_msg
                        # self.print_eval_page(message, choices, ques_num, file_path)
                        # reply = f"审核结果已储存，下一个？[y/n] 如果想倒回上一个记录，请输入[b]"

                        self.evaluated[file_path].append(ques_num)
                        # clear()
                        continue
                        # # r_choice = print_ui(reply)
                        # if r_choice == "y":
                        #     clear()
                        #     continue
                        # elif r_choice == "b":
                        #     memory_dict = self.memory.pop(-1)
                        #     message = memory_dict["message"]
                        #     choices = memory_dict["choices"]
                        #     ques_num = memory_dict["question"]
                        #     file_path = memory_dict["file_path"]
                        #     u_choice = memory_dict["choice"]
                        #     clear()
                        #     print(f"您在回顾问题{ques_num}的选项，您的选择是：{u_choice}，请做出新的选择：")
                        #     self.print_eval_page(message, choices, ques_num, file_path)
                        #     clear()
                        # else:
                        #     clear()
                        #     self.eval_begin()
                    else:
                        self.print_eval_page(message, choices, ques_num, file_path)
                        reply = f"审核结果已储存，下一个？[y/n] 如果想倒回上一个记录，请输入[b]"

                        self.evaluated[file_path].append(ques_num)
                        clear()
                        r_choice = print_ui(reply)
                        if r_choice == "y":
                            clear()
                            continue
                        elif r_choice == "b":
                            memory_dict = self.memory.pop(-1)
                            message = memory_dict["message"]
                            choices = memory_dict["choices"]
                            ques_num = memory_dict["question"]
                            file_path = memory_dict["file_path"]
                            u_choice = memory_dict["choice"]
                            clear()
                            print(f"您在回顾问题{ques_num}的选项，您的选择是：{u_choice}，请做出新的选择：")
                            self.print_eval_page(message, choices, ques_num, file_path)
                            clear()
                        else:
                            clear()
                            self.eval_begin()
        done_msg = f"您已经完成了当前分配的所有审核内容，请等待程序保存数据退出后，请将程序或数据文件打包发送至联系人 刘海江：\n" \
                   f"联系方式：\n" \
                   f"-> QQ：1103478225\n" \
                   f"-> 邮箱：alecliu@ontoweb.wust.edu.cn\n" \
                   f"-> 微信：onlyking1103478225\n\n" \
                   f"感谢您的配合，祝您生活愉快！\n" \
                   f"按任意键结束..."
        return done_msg

    def print_eval_page(self, message, choices, ques_num, file_path):
        user_choice = print_ui(message)
        user_choices = user_choice.split(" ")
        print(user_choices)
        if len(user_choices) == 0:
            clear()
            print(f"没有做出选择？请从下列选项中选一个 {str(choices.keys())}")
            self.print_eval_page(message, choices, ques_num, file_path)
        elif len(user_choices) > 1 and ques_num in self.multiple_choices_questions:
            if ((args.lang == "English" and ques_num in (7, 304)) or (
                    args.lang == "Chinese" and ques_num in (7, 227))) and len(user_choices) > 5:
                clear()
                print(f"多选题要求最多5个选项，请检查选项是否多余，若回答中超过5个则视为未完成测试！")
                print(len(user_choices))
                self.print_eval_page(message, choices, ques_num, file_path)
        elif self.args.intellect == "no" and len(user_choices) > 1 and ques_num not in self.multiple_choices_questions:
            clear()
            print(f"该题目不是多选题，请检查输入！")
            self.print_eval_page(message, choices, ques_num, file_path)
        for user_choice in user_choices:
            self.user_choice_handler(user_choice=user_choice, file_path=file_path, ques_num=ques_num, choices=choices,
                                     message=message)

    def user_choice_handler(self, user_choice, file_path, ques_num, choices, message):
        if "/" in user_choice:
            with open("human_eval/ambiguous_choices.txt", "a", encoding="utf-8") as um_log:
                um_log.write(f"{file_path}--{ques_num}--{user_choice}\n")
        else:
            if user_choice.isnumeric() and int(user_choice) in choices.keys():
                if int(user_choice) == -1:
                    self.memory.append(
                        {"choice": self.fail_code, "question": ques_num, "message": message, "choices": choices,
                         "file_path": file_path})
                else:
                    self.memory.append(
                        {"choice": user_choice, "question": ques_num, "message": message, "choices": choices,
                         "file_path": file_path})
                return
            else:
                clear()
                print(f"没有该选项: {user_choice}, 选项有：{choices}, 请从下列选项中选一个 {str(choices.keys())}")
                self.print_eval_page(message, choices, ques_num, file_path)

    def process_choice(self):
        for memory in self.memory:
            choice = memory["choice"]
            ques_num = memory["question"]
            self.qa_score[choice][ques_num] += 1


def welcome_page(args):
    message = "Hi, there! You are participating a human evaluation for Haijiang Liu's doctorate paper called 'LLM Value \n" \
              "Investigation Project' (current name).\n\n" \
              "Thank you so much for the assistance. Here's your assigment:\n\n" \
              "You are about to observe some of the respeonses generated by Large Language Models (LLM). These responses\n" \
              " are original, so they might reads weird.\n" \
              "I'm asking you to match the response with the choices, which I'll also present to you. This is \n" \
              "paricularly hard for machine, so that's why I asked you to help.\n" \
              "You might get a free meal, drinks or other form of REWARD after you finish! ^_^\n\n" \
              "PLEASE conduct the evaluation in the following steps:\n" \
              "1. You need to read the notes for each question, so that the judgment of every evaluation are the same.\n" \
              "2. I recommand you to read the response first, then see the question and corresponding choices. This will\n" \
              "   make sure you won't have a expectation of the answer and affect your judgement on the model responses.\n" \
              "3. Then you should match the response with the given choices. The program will record the choice. And \n" \
              "   then you can move to the next response.\n" \
              "4. PLEASE remenber, you are not the one who answer these questions, the models are. DO NOT mix your value\n" \
              "   and try to be as objective as possible.\n\n" \
              "By the way, I recommend you to evaluate in the original language for higher accuracy, but if you don't\n" \
              "like it, feel free to change. We support many languages:" + str(supporting_languages)
    clear()
    choices = print_ui(message)
    clear()
    if choices == "a":
        welcome_page(args)
    elif choices == "b":
        welcome_page_chinese(args)
    elif choices == "c":
        iqa_file_dict, qa_score, evaluated, fail_code = read_files(args=args)
        # print(evaluated)
        human_eval = HumanEval(iqa_file_dict=iqa_file_dict, qa_score=qa_score, evaluated=evaluated, fail_code=fail_code,
                               args=args)
        human_eval.eval_begin()
    else:
        print(f"No choice for {choices}, please choose from {str(supporting_languages.keys())}")
        welcome_page(args)


def welcome_page_chinese(args):
    message = "您好！您正在参与对刘海江博士论文 ‘LLM Value Investigation Project’（暂定名）的人工评估。\n\n" \
              "非常感谢您的帮助。以下是您的任务：\n\n" \
              "您将观察由大型语言模型（LLM）生成的一些回答。这些回答是模型的原始输出，所以可能读起来很奇怪。\n" \
              "请您帮忙将这些回答与选项匹配起来，我也会将这些问题和选项展示给您。这对机器来说特别难，所以我请您帮忙。\n" \
              "完成后，您可能会得到一顿免费午餐、饮料或其他形式的奖励！^_^\n\n" \
              "请按以下步骤进行评估：\n" \
              "1. 您需要阅读每个问题的注释，以便每次评估的判断都相同。\n" \
              "2. 我建议您先阅读答案，然后再看问题和相应的选项。这样可以确保您不会对答案有预期，从而影响您对示范答案的判断。\n" \
              "3. 然后，您应该将答案与给出的选项进行匹配。程序将记录您的选择。然后您可以进入下一个答案。\n" \
              "4. 请记住，您不是回答这些问题的人，模型才是。不要混入您的价值去向并尽量保持客观。\n\n" \
              "顺便说一下，我建议您使用原文进行评估，以提高准确性。如果您不喜欢当前语言，请随时更改。\n 我们支持多种语言：" + str(
        supporting_languages)
    choices = print_ui(message)
    clear()
    if choices == "a":
        welcome_page(args)
    elif choices == "b":
        welcome_page_chinese(args)
    elif choices == "c":
        iqa_file_dict, qa_score, evaluated, fail_code = read_files(args=args)
        # for key in iqa_file_dict.keys():
        #     print(f"{key}: len: {len(iqa_file_dict[key])}")
        human_eval = HumanEval(iqa_file_dict=iqa_file_dict, qa_score=qa_score, evaluated=evaluated, fail_code=fail_code,
                               args=args)
        human_eval.eval_begin()
    else:
        print(f"没有该选项: {choices}, 请从下列选项中选一个 {str(supporting_languages.keys())}")
        welcome_page_chinese(args)


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
    if args.intellect == "no":
        with open(args.output_dir + os.sep + "data_processing.log", "r", encoding="utf-8") as log:
            logs_ = log.read()
            logs = logs_.split(" is complete ...")
        file_log_dict = {}
        for _log in logs:
            if _log != "\n" and "The two responses are the same:" not in _log:
                if "investigate" in _log:
                    file_log_dict[_log.split("investigate/")[1]] = _log.split("investigate/")[0]
                else:
                    file_log_dict[_log.split("data analysis for ")[1]] = _log.split("data analysis for ")[0]
    else:
        file_log_dict = {}
        file_log_dict[args.output_dir] = sorted(get_files(args.output_dir))
    # Cipher questions
    qa_list, choice_num_list, question_list = get_choice_num_list(lang=args.lang, questionnaire_path=args.questionnaire)

    # Cipher instructions
    qi_dict = get_instructions(args.instruction_path)

    # Cipher answers
    iqa_file_dict = {}
    if args.intellect == "no":
        for file_path in file_log_dict.keys():
            file_path_win = file_path.replace("/", "\\")
            answers = prepare_answers(choices_list=choice_num_list, output_file=file_path_win)
            file_logs = file_log_dict[file_path]
            log_lines = file_logs.split("\n")[:-1]
            iqa_pairs = []
            for log_line in log_lines:
                pattern = re.compile(r"quenstion .*\d ", re.I)
                ques_num = re.findall(pattern, log_line)
                if len(ques_num) != 0:
                    ques_num = int(ques_num[0].replace("quenstion ", ""))
                    iqa_pairs.append((ques_num, qi_dict[ques_num], question_list[ques_num - 1], qa_list[ques_num - 1],
                                      answers[ques_num - 1]))
            iqa_file_dict[file_path] = iqa_pairs
    else:
        for file in file_log_dict[args.output_dir]:
            log_file = file.split(".txt")[0] + ".log"
            with open(log_file, 'r', encoding="utf-8") as log:
                random_index = log.read().split('\n')
            iqa_pairs = []
            answers = prepare_answers(choices_list=choice_num_list, output_file=file)
            for ques_num in range(len(answers)):
                if str(ques_num) in random_index:
                    iqa_pairs.append((ques_num + 1, qi_dict[ques_num + 1], question_list[ques_num], qa_list[ques_num],
                                      answers[ques_num]))
            iqa_file_dict[file] = iqa_pairs

    if os.path.exists(args.data_storage):
        qa_score = pd.read_csv(args.data_storage, index_col=0)
    else:
        columns = range(max(choice_num_list) + 2)
        columns = list(map(str, columns))
        qa_score = pd.DataFrame(columns=columns, index=range(len(choice_num_list) + 1), data=0)
    if args.intellect == "no":
        out_dir = "human_eval"
    else:
        out_dir = "gpt_judge"
    if os.path.exists(out_dir + f"\\{args.lang}-evaluated.json"):
        with open(out_dir + f"\\{args.lang}-evaluated.json", "r", encoding="utf-8") as history_file:
            evaluated = json.load(history_file)
    else:
        evaluated = {}
    return iqa_file_dict, qa_score, evaluated, max(choice_num_list) + 1


def clear():
    # for windows
    if name == 'nt':
        _ = system('cls')
    # for mac and linux(here, os.name is 'posix')
    else:
        _ = system('clear')

    # print out some text


def print_ui(messages):
    outlier_upper = "=" * 120
    outlier_lower = "=" * 120
    print(outlier_upper)
    print(messages)
    print(outlier_lower)
    choices = input("Your choices: ")
    return choices


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
        default="outputs\\Chinese_chatglm",
        help="model response dir"
    )
    parser.add_argument(
        "--data_storage",
        type=str,
        default="human_eval\\qa_scores.csv",
        help="data storage path, direct to a excel file"
    )
    parser.add_argument(
        "--questionnaire",
        type=str,
        default="survey_data\\sample\\",
        help="Questionnaire dir path")
    parser.add_argument(
        "--instruction_path",
        type=str,
        default="survey_data\\introse\\CN_intellect_instructions.txt",
        help="instruction file path")
    parser.add_argument(
        "--intellect",
        type=str,
        default="no",
        help="whether do intellect measure"
    )
    return parser


if __name__ == '__main__':
    parser = argument_parser()
    args = parser.parse_args()
    if not os.path.exists(args.data_storage):
        os.makedirs(args.data_storage)
    args.data_storage += "\\extra_score.csv"
    clear()
    welcome_page_chinese(args=args)
