# LLM Trees - Decision Trees through LLMs
![python](https://img.shields.io/badge/python-3.12.13%2B-blue)

## SUMMARY
Replication of Oh LLM, I’m Asking Thee, Please Give Me a Decision Tree”: Zero-Shot Decision Tree Induction and Embedding with Large Language Models (KDD conference) paper for DS8008 Class.
- We prompt Gemini 2.5 Flash with informationa about a dataset, such as features, and ask it to generate decision tree logic internally.
- It also then takes that internal logic and creates a function that can be used as a classifier.
- The logic of these functions are then used to evaluate the performance of of these decision trees vs. randomly guessing.
- Each dataset is used to prompt the LLM 5 times, to generate 5 different decision trees.

Original Paper Repository: https://github.com/ml-lab-htw/llm-trees
## 📂 Folder Structure
```
DS8008_LLM_TREE
├──data
│   ├── data_sets
│   └── Gemini_outputs
├──src
│   ├── Evaluators
│   │   ├──Embeddings
│   │   │   ├── config.py
│   │   │   ├── emb_generator_bankruptcy.py
│   │   │   ├── emb_generator_boxing1.py
│   │   │   ├── emb_generator_boxing2.py
│   │   │   ├── emb_generator_colid.py
│   │   │   ├── emb_generator_credit.py
│   │   │   └── embedding_eval.py
│   │   ├──model_extractors
│   │   │   ├── bankruptcy_models.py
│   │   │   ├── boxing1_models.py
│   │   │   ├── boxing2_models.py
│   │   │   ├── colic_models.py
│   │   │   └── credit_models.py
│   │   ├── bankruptcy.py
│   │   ├── boxing1.py
│   │   ├── boxing2.py
│   │   ├── colic.py
│   │   ├──creditcore.py
│   │   ├── emb_eval_looper.py
│   │   ├── evaluate_bankruptcy.py
│   │   ├── evaluate_boxing1.py
│   │   ├── evaluate_boxin2.py
│   │   ├── evaluate_colic.py
│   │   └── evaluate_creditcore.py
│   ├── Figures
│   │   ├── func_example_full.png
│   │   ├── llm_flow.png
│   │   ├── sample_output.png
│   │   ├── sample_output2.png
│   │   └── sample_return_format.png
│   ├── dt_extractor_colab.ipynb
│   ├── gemini_prompter.py
    └── plotter.py
├──.gitignore
├── LICENSE
├── LLM_Trees_Abstract.pdf
├──LLM_Trees_project_report.ipynb
├── README.md
└── requirements.txt
```

## Required Environment
- python >= 3.12.13
``` python
pip install -r requirements.txt
```
* You will also need to install Ollama software from, https://ollama.com/download/windows
* In the env terminal after running requirements.txt, which also installs ollama python package to env, you will need to run the following commands to pull the models to your env.
```bash
ollama pull gpt-oss:20b
ollama pull qwen3:14B
ollama pull gemma3:12b
ollama pull mistral-small3.2:24b
```

## Evaluate LLM Trees
This is done using the [project report notebook](LLM_Trees_project_report.ipynb) which you can easily follow along to test the functions that we extracted.

**[NOTE]** To test your own functions that you extract, you will need to edit them yourself, and create an evaluation function for them. You can refer to the scripts in `src/Evaluators` for examples.

