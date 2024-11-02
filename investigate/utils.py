import random
import re
from typing import List, Optional

import torch
import transformers
from transformers import GenerationConfig, LogitsProcessor, LogitsProcessorList

from conversation import get_conv_template

lang_to_country = {
    "English": ["United States"], "Chinese": ["中国"], "en_Chinese": ["China"], "zh_English": ["美国"],
    "Chinese_rephrased": ["中国"], "English_rephrased": ["United States"],
    "en_Chinese_rephrased": ["China"], "zh_English_rephrased": ["美国"]
}
US_states = ["Alabama", "Alaska", "Arizona", "Arkansas", "California", "Colorado", "Connecticut", "Delaware", "Florida",
             "Georgia", "Hawaii", "Idaho", "Illinois", "Indiana", "Iowa", "Kansas", "Kentucky", "Louisiana", "Maine",
             "Maryland", "Massachusetts", "Michigan", "Minnesota", "Mississippi", "Missouri", "Montana", "Nebraska",
             "Nevada", "New Hampshire", "New Jersey", "New Mexico", "New York", "North Carolina", "North Dakota",
             "Ohio", "Oklahoma", "Oregon", "Pennsylvania", "Rhode Island", "South Carolina", "South Dakota",
             "Tennessee", "Texas", "Utah", "Vermont", "Virginia", "Washington", "West Virginia", "Wisconsin", "Wyoming"]
zh_US_states = ["阿拉巴马州", "阿拉斯加州", "亚利桑那州", "阿肯色州", "加利福尼亚州", "科罗拉多州", "康涅狄格州",
                "特拉华州", "佛罗里达州",
                "佐治亚", "夏威夷", "爱达荷", "伊利诺伊", "印第安纳", "爱荷华", "堪萨斯", "肯塔基", "路易斯安那",
                "缅因州",
                "马里兰州", "马萨诸塞州", "密歇根州", "明尼苏达州", "密西西比州", "密苏里州", "蒙大拿州",
                "内布拉斯加州",
                "内华达州", "新罕布什尔州", "新泽西州", "新墨西哥州", "纽约州", "北卡罗来纳州", "北达科他州",
                "俄亥俄州", "俄克拉何马州", "俄勒冈州", "宾夕法尼亚州", "罗得岛州", "南卡罗来纳州", "南达科他州",
                "田纳西州", "得克萨斯州", "犹他州", "佛蒙特州", "弗吉尼亚州", "华盛顿州", "西弗吉尼亚州", "威斯康星州",
                "怀俄明州"]
CH_states = ["河北省", "山西省", "辽宁省", "吉林省", "黑龙江省", "江苏省", "浙江省", "安徽省", "福建省", "江西省",
             "山东省", "河南省",
             "湖北省", "湖南省", "广东省", "海南省", "四川省", "贵州省", "云南省", "陕西省", "甘肃省", "青海省",
             "台湾省"]
en_CH_states = ["Hebei Province", "Shanxi Province", "Liaoning Province", "Jilin Province", "Heilongjiang Province",
                "Jiangsu Province", "Zhejiang Province", "Anhui Province", "Fujian Province", "Jiangxi Province",
                "Shandong Province", "Henan Province", "Hubei Province", "Hunan Province", "Guangdong Province",
                "Hainan Province", "Sichuan Province", "Guizhou Province", "Yunnan Province", "Shaanxi Province",
                "Gansu Province", "Qinghai Province", "Taiwan Province"]
country_to_states = {"United States": US_states, "中国": CH_states, "China": en_CH_states, "美国": zh_US_states}

self_aware_keywords = r"As an artificial intelligence assistant|As an AI language model|作为一个人工智能助手|作为一个大语言模型"


def prepare_answers(output_file, choices_list):
    with open(output_file, "r",
              encoding="utf-8") as ans:
        answers_raw = ans.read()
        answers_raw = answers_raw.split("\n")

    answers = []

    for i in range(len(answers_raw)):
        if answers_raw[i].startswith("Q"):
            answer = answers_raw[i]
            answers.append(answer)
        elif answers_raw != "\n":
            answers[-1] += answers_raw[i]

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
                choices_text_list = re.split(':|,', choices_text[1])
                for i in range(0, len(choices_text_list), 2):
                    choice_text = choices_text_list[i + 1].strip()
                    choice_num = int(choices_text_list[i])
                    choices_text_dict[choice_num] = choice_text
            choices = _choices.split("[")[1]
            qa_list.append(choices_text_dict)
            choice_num_list.append(int(choices))

            question = q.split("]")[1]
            question_list.append(question)
    return qa_list, choice_num_list, question_list


def shuffle_choices(wvs_input):
    in_prompt = wvs_input[:wvs_input.find("Request: ") + len("Request: ")]
    if wvs_input.find("- ") != -1:
        ques_body = wvs_input[wvs_input.find("Request: ") + len("Request: "): wvs_input.find("- ")]
        choice_body = ''
        choices = wvs_input[wvs_input.find("- ") + len("- "):]
        choice_list = choices.split(", ")
        random.shuffle(choice_list)
        for c in choice_list:
            choice_body = choice_body + ", " + c
        new_wvs_input = in_prompt + ques_body + choice_body
    else:
        new_wvs_input = wvs_input
    return new_wvs_input


def build_chat_input(model, tokenizer, input: List[dict], history, conv, is_base=False):
    def _parse_messages(history, query):
        if history is None:
            history = []
        for i, (old_query, response) in enumerate(history):
            conv.append_message(conv.roles[0], old_query)
            conv.append_message(conv.roles[1], response)
        conv.append_message(conv.roles[0], query)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        return prompt

    prompt = _parse_messages(history, input)
    if is_base:
        prompt += "Answer: "
    input = tokenizer.encode(prompt)
    return torch.IntTensor([input]).to(model.device)


def chat(model, model_name, tokenizer, inputs, history: List,
         generation_config: Optional[GenerationConfig] = None, logits_processor=None):
    conv = get_conv_template(model_name)
    msgs = []
    if "claude" in model_name:
        for i, (old_query, response) in enumerate(history):
            msgs.append({"role": conv.roles[0], "content": old_query})
            msgs.append({"role": conv.roles[1], "content": response})
        msgs.append({"role": conv.roles[0], "content": inputs})
        response_msg = model["model"].messages.create(
            model=model["model_id"],
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            top_k=generation_config.top_k,
            system=conv.system_message,
            messages=msgs
        )
        response = response_msg.content[0].text
        history = history + [(inputs, response)]
    elif "gpt" in model_name:
        msgs.append({"role": "system", "content": conv.system_message})
        for i, (old_query, response) in enumerate(history):
            msgs.append({"role": conv.roles[0], "content": old_query})
            msgs.append({"role": conv.roles[1], "content": response})
        msgs.append({"role": conv.roles[0], "content": inputs})
        response_msg = model["model"].chat.completions.create(
            model=model["model_id"],
            max_tokens=generation_config.max_new_tokens,
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            messages=msgs
        )
        response = response_msg.choices[0].message.content
        history = history + [(inputs, response)]
    else:
        input_ids = build_chat_input(model, tokenizer, inputs, history, conv,
                                     is_base=True if "base" in model_name else False)
        if logits_processor is None:
            logits_processor = LogitsProcessorList()
        logits_processor.append(InvalidScoreLogitsProcessor())
        generation_config = generation_config if generation_config is not None else model.generation_config
        outputs = model.generate(
            input_ids=input_ids, generation_config=generation_config, logits_processor=logits_processor,
            max_new_tokens=generation_config.max_new_tokens, pad_token_id=tokenizer.eos_token_id
        )
        response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
        history = history + [(inputs, response)]
    return response, history


def ccsv_generator(model, tokenizer, inputs, generation_config, logits_processor=None):
    input_ids = tokenizer.encode(inputs)
    input_ids = torch.tensor([input_ids]).to(model.device)

    if logits_processor is None:
        logits_processor = LogitsProcessorList()
    logits_processor.append(InvalidScoreLogitsProcessor())
    generation_config = generation_config if generation_config is not None else model.generation_config
    generation_config.logits_processor = logits_processor
    outputs = model.generate(input_ids, generation_config=generation_config)
    response = tokenizer.decode(outputs[0][len(input_ids[0]):], skip_special_tokens=True)
    input_ids = input_ids.to("cpu")

    critque_prompt = "Question: Critique the Al model's response and identify ways in which it lacks diversity. " \
                     "Provide a suggestion on how to improve theanswer.\n" \
                     "Critique and suggestion:"
    critque_input = "User: " + inputs + '\n' + "Response: " + response + "\n" + critque_prompt
    cinput = tokenizer.encode(critque_input)
    cinput_ids = torch.LongTensor([cinput]).to(model.device)
    outputs = model.generate(cinput_ids, generation_config=generation_config)
    cresponse = tokenizer.decode(outputs[0][len(cinput_ids[0]):], skip_special_tokens=True)
    cinput_ids = cinput_ids.to("cpu")

    rewrite_prompt = "User: " + inputs + '\n' + "Response: " + response + "\n" + \
                     "Here's a list of critiques and suggestions:" + cresponse + \
                     "Question: Rewrite the Al model's response to the user's question based on the Critique and suggestions above." \
                     "Please ensure that the user's specific question is answered. \nRevised Al model response: "
    reinput = tokenizer.encode(rewrite_prompt)
    reinput_ids = torch.LongTensor([reinput]).to(model.device)
    outputs = model.generate(reinput_ids, generation_config=generation_config)
    reresponse = tokenizer.decode(outputs[0][len(reinput_ids[0]):], skip_special_tokens=True)
    reinput_ids = reinput_ids.to("cpu")

    return reresponse


class InvalidScoreLogitsProcessor(LogitsProcessor):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores.zero_()
            scores[..., 5] = 5e4
        return scores
