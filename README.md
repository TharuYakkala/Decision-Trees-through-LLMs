# LLM Trees - Decision Trees through LLMs
![python](https://img.shields.io/badge/python-3.12.13%2B-blue)

## OVERVIEW

This repository is on the *replication* of the research paper: 

> #### "Oh LLM, I'm Asking Thee, Please Give Me a Decision Tree": Zero-Shot Decision Tree Induction and Embedding with Large Language Models (Knauer et al., 2025)

The original research paper can be found here: [link to original paper](https://dl.acm.org/doi/10.1145/3711896.3736818)  

Original paper repository: [link to original repo](https://github.com/ml-lab-htw/llm-trees)  

<br>

The original paper implements the following:  

**1. Zero-shot Decision Tree Induction**  
- Prompting LLM(s) using zero-shot prompting to derive a decision tree classifier by *only* passing features of the dataset (no training data passed to LLM)  

**2. Embedding Induction**  
- Generate decision trees whose nodes are used to create binary feature vectors as inputs for neural network models  
  
<br> 

The first half of this readme covers the original research paper while the latter half covers our implementation.

## The Research Paper: Motivation, Methdology, Evaluation
In this section, we briefly discuss the <u> original research paper </u> by Knauer et al. (2025) and explain key insights and motivations.  

**Motivation**  
The *motivation* behind the paper comes from **knowledge distillation** of LLMs (their "world knowledge") and the idea that data can be scarce or propriety. This paper also draws from **in-context learning** and **transfer learning**. The key idea is to find a way to use the "world knowledge" that large language models have, which have been trained on an enormous amount of data from the internet, and derive *models* (in this case, decision trees), **without any training data** ever passed to the LLM. This alleviovercomes the issue of privacy concerns with proprietary data and demonstrates how LLMs can still be leveraged for small datasets that would otherwise be difficult to train in models.

> Exercept from the paper: "we present the first approach to apply state-of-the-art LLMs for zero-shot model generation using in-context learning, i.e., we show how LLMs can build intrinsically interpretable trees without access to pretrained model weights and without any training data" (Knauer et al., 2025)  

Domains of study: large language models, zero-shot prompting, in-context learning, knowledge distillation, transfer learning, data scarcity, intrinsic model induction   

<br>  

**Methdology**  
*Part 1: Zero-shot Decision Tree Induction*  
Authors ask an LLM to generate a decision tree using only the feature names (column headers); no actual data values, no examples (zero-shot prompting). Then the LLM is used again to convert text-based decision tree into a Python function to be able to make predictions on data.
→ LLM Models Used: Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4o, GPT-o1  
→ Baseline Models Used: BSS, OCTs, AutoGluon, Auto-Prognosis, TabPFN

*Part 2: Embedding Induction*  
The generated decision tree's structure is used to create an embedding of each data point. For a given sample, every internal decision node in the tree is a binary (0 or 1) value. The authors generate 5 diverse trees per LLM (temperature used to control variety) and concatenate all the node truth values into a single vector that is fed into a smal NN with the original features for classification.   
→ LLM Models Used: Claude 3.5 Sonnet, Gemini 1.5 Pro, GPT-4o, GPT-o1  
→ Baseline Models Used: MLP with no embedding, Random trees embeddings, Extra trees embeddings, Random forests embeddings, XGboost embeddings 

**Evaluation**  
Metrics used: macro F1-score, balanced accuracy

<br> 

## OUR PROJECT  
**Our project replicates the original research paper by implementing <u>both</u> (1) Decision Tree Induction and (2) Embedding Induction.  
However, we use <u>different LLM models and only 5 datasets</u> in comparison.**  

Datasets Used:  
1. bankruptcy.py
2. boxing1.py
3. boxing2.py
4. colic.py
5. creditscore.py  

LLM Models Used (Ollama):  
1. gemma3
2. gpt_oss
3. mistral_small3
4. qwen3  

Part 1 Baseline Models Used:  
- Autogluon 
- TabPFN
  
Part 2 Baseline Models Used:  
- MLP
- RandomTrees
- XGBoost
- ..

<br>  

**Note:** You can find a table of implementation comparison in *LLM_trees_project_report.ipynb* to see how our implementation differs/matches the original paper  
  

### To begin, first look at the folder structure below. 


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


## SUMMARY
Replication of Oh LLM, I’m Asking Thee, Please Give Me a Decision Tree”: Zero-Shot Decision Tree Induction and Embedding with Large Language Models (KDD conference) paper for DS8008 Class.
- We prompt 4 Open Source Models from Ollama with informationa about a dataset, such as features, and ask it to generate decision tree logic internally.
- It also then takes that internal logic and creates a function that can be used as a classifier.
- The logic of these functions are then used to evaluate the performance of of these decision trees in classification and embedding extraction.
- Each dataset is used to prompt the LLM 5 times, to generate 5 different decision trees. This is done twice, once for decision tree induction method, and another time for decision tree embedding method.

Original Paper Repository: https://github.com/ml-lab-htw/llm-trees