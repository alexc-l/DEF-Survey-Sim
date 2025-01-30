# Towards Realistic Evaluation of Cultural Value Alignment in Large Language Models: Diversity Enhancement for Survey Response Simulation
![pipeline.png](docs%2Fpipeline.png)
## Overview
This repository contains the implementation of our paper ["Towards Realistic Evaluation of Cultural Value Alignment in Large Language Models: Diversity Enhancement for Survey Response Simulation"]().

We present a framework for evaluating how well Large Language Models (LLMs) align with cultural values through enhanced survey response simulation. Our approach incorporates diversity enhancement techniques to ensure robust evaluation.

## Dataset
The dataset used in this research is available on [HuggingFace](https://huggingface.co/datasets/alec-x/Diversity-enhanced-survey-simulation). It includes carefully curated questions from major social surveys targeting value preferences across different cultures.

## Installation
```shell
pip install -r requirements.txt
```
# Usage
## 1. Survey Response Generation
Run the simulation using an LLM with the following configuration options:
```shell
python main_test.py \
  --prompt_path <path_to_prompt> \
  --questionnaire <survey_data_directory> \
  --model_name <model_path_or_name> \
  --model_name_short $model_name_short \
  --model_class $model_class \
  --output_dir <output_directory> \
  --generate_kwargs $generate_kwargs \
  --case_num <number_of_cases> \
  --diverse_plan [prompt|config|pc|all] \
  --lang [Chinese|English] \
  --resume_epoch $resume_epoch \
  --resume_input $resume_input
```
### Key Parameters
| Parameter        | Description                | Options                                                  |
|------------------|----------------------------|----------------------------------------------------------|
| model_name_short | Supported model type       | baichuan2-chat, chatglm2, mistral, llama-3, llama-3-base |
| model_class      | Model loading method       | AutoModelForCausalLM, Claude, GPT                        |
| lang             | Target language/culture    | Chinese, English                                         |
| diverse_plan     | Diversity enhancement type | prompt, config, pc, all                                  |

Full parameter list available in [evaluation.sh](investigate/scripts/evaluation.sh)

## 2. Evaluation
### Generate Preference Distributions
```shell
python auto_analyze.py \
    --lang [Chinese|English] \
    --output_dir <generation_output> \
    --data_storage <distribution_output.csv> \
    --questionnaire <survey_data_directory>
```
Full parameter list available in [auto_analysis.sh](evaluation/scripts/auto_analysis.sh)

### Human Evaluation
```shell
python human_eval.py \
    --lang [Chinese|English] \
    --output_dir <generation_output> \
    --data_storage <distribution_output> \
    --questionnaire <survey_data_directory> \
    --instruction_path <human_instruction.txt>
```
Full parameter list available in [human_eval_start.cmd](evaluation/scripts/human_eval_start.cmd).

### Insensitivity Measurement
```shell
python intellect_eval.py \
    --lang [Chinese|English] \
    --output_dir <generation_output> \
    --data_storage <distribution_output> \
    --questionnaire <survey_data_directory> \
    --instruction_path <human_instruction.txt> \
    --intellect yes
```
Full parameter list available in [gpt_jugde.cmd](evaluation/scripts/gpt_jugde.cmd)

## Visualization
Analyze and visualize results using our Jupyter [notebook](evaluation/Result_figures.ipynb).

## Results Interpretation
Our evaluation framework measures alignment across four key metrics:
- Preference distribution alignment
- Cultural sensitivity
- Demographic bias
- Response quality

# Contact
**Technical questions**: Create a GitHub Issue

**Research inquiries**: alecliu@ontoweb.wust.edu.cn

# Citation
If you use this code or our findings in your research, please cite:
```bibtex
Update when available.
```
