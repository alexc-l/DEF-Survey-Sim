import argparse
import json
import os
import re
import time

import anthropic
# import anthropic
import numpy
import openai
# import openai
import torch
from tqdm.auto import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM, GenerationConfig
from auto_gptq import AutoGPTQForCausalLM
import utils
from bloom_inference_server.constants import HF_ACCELERATE, DS_ZERO, HF_CPU, DS_INFERENCE
from utils import lang_to_country, country_to_states, shuffle_choices


def perpare_wvs_inputs(prompt, lang, ques_dir, mode):
    questionaire_path = ques_dir + lang + "_questionaires.txt"
    with open(questionaire_path, "r", encoding="utf-8") as q:
        ques = q.read()
        questions = ques.split("\n")
    inputs = []

    if mode in ["prompt", 'pc']:
        state_list = country_to_states[lang_to_country[lang][0]]
        state_num = numpy.random.randint(low=0, high=len(state_list))
        lang_and_states = lang_to_country[lang][0] + " " + state_list[state_num]
    else:
        lang_and_states = lang_to_country[lang][0]

    for q in questions:
        input_data = prompt.replace("[COUNTRY]", lang_and_states)
        q = q.replace("[COUNTRY]", lang_to_country[lang][0])
        # print(q)
        _choices = q.split("]")[0]
        choices = _choices.split("[")[1]
        question = q.split("]")[1]
        input_choice = input_data.replace("[NUM]", choices)
        input_choice_question = input_choice.replace("[QUESTION]", question)
        inputs.append(input_choice_question)
    return inputs


def argument_parser():
    parser = argparse.ArgumentParser(description='LLM Value Investigation args', epilog='Information end')
    parser.add_argument(
        '--ccsv_test',
        type=bool, default=False, required=False, help='Ablation study on CCSV'
    )
    parser.add_argument(
        '--resume_epoch',
        type=int, default=0, required=False, help='Tested epochs from checkpoint'
    )
    parser.add_argument(
        '--resume_input',
        type=int, default=0, required=False, help='Tested input from checkpoint'
    )
    parser.add_argument(
        "--prompt_path",
        type=str,
        required=True,
        help="Prompt file path, direct to a txt file")
    parser.add_argument(
        "--resume_count",
        type=int,
        default=0,
        help="resume from question number what?"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="English",
        help="questionnaire language")
    parser.add_argument(
        "--questionnaire",
        type=str,
        required=True,
        help="Questionnaire dir path")
    parser.add_argument(
        "--case_num",
        type=int,
        default=10,
        help="investigate case number")
    parser.add_argument(
        "--test_per_case",
        type=int,
        default=10,
        help="test number to lessen position bias")
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="output dir")
    parser.add_argument(
        "--diverse_plan",
        type=str,
        required=True,
        help="Diverse plan: ['prompt', 'config', 'pc', 'all'] memory can be altered by max_memory in generation kwargs"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        required=False,
        help="Distribution training"
    )
    group = parser.add_argument_group(title="model")
    group.add_argument(
        "--deployment_framework",
        type=str,
        choices=[HF_ACCELERATE, DS_INFERENCE, DS_ZERO, HF_CPU],
        default=HF_ACCELERATE,
    )
    group.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="model name to use",
    )
    group.add_argument(
        "--model_name_short",
        type=str,
        required=True,
        help="model name for prompt {"
             "'baichuan2': 'baichuan2-chat', "
             "'chatglm2': 'chatglm2', "
             "'mistral-7B': 'mistral', "
             "'llama3-instruct-8B': 'llama-3', "
             "'llama3-base-8B': 'llama-3-base', "
             "'Dolphin-2.9.1-llama-3-8b': 'llama-3', "
             "'mixtral-8*7B': 'mistral', "
             "'llama3-chinese': 'llama3', "
             "}",
    )
    group.add_argument(
        "--model_class",
        type=str,
        required=True,
        help="model class to use",
    )
    group.add_argument(
        "--dtype", type=str, required=True, choices=["bf16", "fp16", "int8", "fp32"], help="dtype for model"
    )
    group.add_argument(
        "--generate_kwargs",
        type=str,
        help="generate parameters. look at https://huggingface.co/docs/transformers/v4.21.1/en/main_classes/text_generation#transformers.generation_utils.GenerationMixin.generate to see the supported parameters",
    )
    group.add_argument("--max_input_length", type=int, help="max input length")
    group.add_argument("--max_batch_size", type=int, help="max supported batch size")
    return parser


def parse_args(parser):
    args = parser.parse_args()
    if args.generate_kwargs is not None:
        with open(args.generate_kwargs, "r", encoding="utf-8") as config:
            args.generate_kwargs = json.load(config)

    model_name = args.model_name.split(os.sep)[-1]
    resume_generate_config = args.output_dir + os.sep + model_name + os.sep + str(args.resume_epoch)
    if os.path.exists(resume_generate_config + os.sep + "0_generation_config.json"):
        with open(resume_generate_config + os.sep + "0_generation_config.json", "r", encoding="utf-8") as rconfig:
            args.resume_generate_config = json.load(rconfig)
            print(args.resume_generate_config)
            args.resume = True
    else:
        args.resume = False
    return args


def main():
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)

    parser = argument_parser()
    args = parse_args(parser)

    with open(args.prompt_path, "r", encoding='utf-8') as f:
        prompt = f.read()  # 读取文本
    lang = args.lang
    model_id = args.model_name
    if args.model_class.find("AutoModelForCausalLM") != -1 and "GPTQ" not in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(model_id)
        model.generation_config.max_new_tokens = 128
    elif args.model_class.find("Claude") != -1:
        os.environ["http_proxy"] = "127.0.0.1:7890"
        os.environ["https_proxy"] = "127.0.0.1:7890"
        model = {
            "model_id": model_id,
            "model": anthropic.Anthropic(
                api_key="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"
            )
        }
        tokenizer = None
    elif args.model_class.find("GPT") != -1:
        os.environ["http_proxy"] = "127.0.0.1:7890"
        os.environ["https_proxy"] = "127.0.0.1:7890"
        model = {
            "model_id": model_id,
            "model": openai.OpenAI(
                api_key="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"
            )
        }
        tokenizer = None
    elif "4bit" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_id, lood_in_4bit=True, use_flash_attention=True
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(model_id)
        model.generation_config.max_new_tokens = 128
    elif "GPTQ" in args.model_name:
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            # model_basename="model",
            device_map="auto",
            # use_safetensors=True,
            trust_remote_code=False,
            # use_triton=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        model.generation_config = GenerationConfig.from_pretrained(model_id)
        model.generation_config.max_new_tokens = 128
    else:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id, trust_remote_code=True).cuda()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    max_memory = args.generate_kwargs.pop("max_memory")
    beam_num = args.generate_kwargs["num_beams"]
    if args.diverse_plan not in ["prompt", 'pc']:
        wvs_inputs = perpare_wvs_inputs(prompt=prompt, lang=lang, ques_dir=args.questionnaire, mode=args.diverse_plan)

    with tqdm(total=args.case_num * args.test_per_case) as pbar:
        tested_epoch = 0
        for epoch in range(args.case_num):
            if tested_epoch != args.resume_epoch:
                tested_epoch += 1
                pbar.update()
                continue
            else:
                tested_epoch = args.resume_epoch
            model_name = args.model_name_short
            output_dir = args.output_dir + os.sep + model_name + os.sep + str(epoch)
            if args.diverse_plan in ["prompt", 'pc', 'all']:
                wvs_inputs = perpare_wvs_inputs(prompt=prompt, lang=lang, ques_dir=args.questionnaire,
                                                mode=args.diverse_plan)
            if args.diverse_plan in ["config", "pc", 'all']:
                beam = numpy.random.randint(low=1, high=beam_num)
                args.generate_kwargs["num_beams"] = beam
            wvs_output_dict = {}
            mem_history = []
            if max_memory > 0:
                memory = numpy.random.randint(low=0, high=max_memory)
            else:
                memory = 0
            for test_num in range(args.test_per_case):
                wvs_outputs = []
                # i = 1
                # start_time = time.time()
                # input_count = 0
                with tqdm(total=len(wvs_inputs), desc=f"Answering the {epoch}th survey: ") as sbar:
                    tested_input = 0
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    for i in range(len(wvs_inputs)):
                        wvs_input = wvs_inputs[i]
                        if tested_input < args.resume_input:
                            tested_input += 1
                            sbar.update()
                            continue
                        else:
                            tested_input = args.resume_input
                        new_wvs_input = shuffle_choices(wvs_input)
                        if "base" in args.model_name_short:
                            new_wvs_input += "\nAnswer: "
                        if args.resume:
                            generate_kwargs = args.resume_generate_config
                            args.resume = False
                        else:
                            generate_kwargs = args.generate_kwargs
                            generate_kwargs["memory"] = memory
                            generate_kwargs["max_new_tokens"] = args.generate_kwargs["max_new_tokens"]
                            with open(output_dir + os.sep + str(test_num) + "_generation_config.json", 'w',
                                      encoding='utf-8') as ogf:
                                json.dump(generate_kwargs, ogf, ensure_ascii=False, indent="\t", sort_keys=True,
                                          separators=(',', ':'))
                            memory = generate_kwargs["memory"]
                            generate_kwargs.pop("memory")
                            if generate_kwargs.get("max_length") != None:
                                generate_kwargs.pop("max_length")
                                # generate_kwargs["max_new_tokens"] = 128
                        generate_kwargs = GenerationConfig.from_dict(generate_kwargs)
                        if new_wvs_input.find("[PREVIOUS CHIOCE]") != -1:
                            last_response = wvs_outputs[-1]
                            last_response = last_response[last_response.find("Answer:"):]
                            new_wvs_input.replace("[PREVIOUS CHIOCE]", last_response)
                        if args.ccsv_test:
                            response = utils.ccsv_generator(model, tokenizer, new_wvs_input, generate_kwargs)
                        else:
                            try:
                                start_time = time.time()
                                response, history = utils.chat(
                                    model=model, model_name=model_name, tokenizer=tokenizer,
                                    inputs=new_wvs_input, history=mem_history, generation_config=generate_kwargs
                                )
                            except:
                                if args.model_class.find("GPT") != -1 or args.model_class.find("Claude") != -1:
                                    limit_time = time.time()
                                    time_gap = limit_time - start_time
                                    time_gap = time_gap * 1000 * 1000
                                    rest_time = time_gap % 60
                                    print(f'\nWaiting response for {rest_time}...\n')
                                    time.sleep(rest_time)
                                    start_time = time.time()
                                response, history = utils.chat(
                                    model=model, model_name=model_name, tokenizer=tokenizer,
                                    inputs=new_wvs_input, history=mem_history, generation_config=generate_kwargs
                                )
                            mem_history.append(history[-1])
                        with open(output_dir + os.sep + str(test_num) + "_test_responses.txt", "a",
                                  encoding="utf-8") as o:
                            line = f"Q{i + 1}: {response}\n"
                            o.write(line)

                        mem_history_cleaned = []
                        for men in mem_history:
                            res = men[1]
                            if "Answer: " in res and re.match(utils.self_aware_keywords, res) is None:
                                que = men[0]
                                res = men[1]
                                que_loc = que.find("Request:")
                                que = que[que_loc:]
                                res_start = res.find("Answer:")
                                exp_star = res.find("Explanation:")
                                exp_end = res[exp_star:].find(".") + exp_star
                                res_end = res[exp_star:exp_end].find("\n") + exp_star
                                res_end = max(res_end, exp_end)
                                res = res[res_start:res_end]
                                mem_history_cleaned.append((que, res))
                        mem_history = mem_history_cleaned
                        if len(mem_history) > memory:
                            temp_history = []
                            # memory_distribution = [1.84 / (math.pow(math.log(t), 1.25) + 1.84) for t in
                            # sample_index = numpy.random.choice(a=range(len(temp_history)), size=memory, p=memory_distribution)
                            if memory > 0:
                                sample_index = numpy.random.choice(a=range(len(mem_history)), size=memory)
                                for index in sample_index:
                                    temp = mem_history[index]
                                    temp_history.append(temp)
                                mem_history = temp_history
                            else:
                                mem_history = []
                        wvs_outputs.append(response)
                        sbar.update()
                args.resume_input = 0
                # wvs_output_dict[test_num] = wvs_outputs
                # count = 1
                # with open(output_dir + os.sep + str(test_num) + "_test_responses.txt", "w", encoding="utf-8") as o:
                #     for wvs_output in wvs_output_dict[test_num]:
                #         line = f"Q{count}: {wvs_output}\n"
                #         count += 1
                #         o.write(line)

                pbar.update()


if __name__ == '__main__':
    main()
