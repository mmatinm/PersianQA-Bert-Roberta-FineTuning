
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mmatinm/PersianQA-Bert-XLMRoberta-FineTuning/blob/main/demo.ipynb)
[![XLM-RoBERTa Model](https://img.shields.io/badge/HuggingFace-mpersian_xlm_roberta_large-yellow?logo=huggingface)](https://huggingface.co/mmatinm/mpersian_xlm_roberta_large)
[![ParsBERT Model](https://img.shields.io/badge/HuggingFace-parsbert_question_answering_PersianQA_m-yellow?logo=huggingface)](https://huggingface.co/mmatinm/parsbert_question_answering_PersianQA_m)

# PersianQA Transformer Fine-Tuning

This repository contains scripts, notebooks, and trained model information for fine-tuning **ParsBERT** and **XLM-RoBERTa** on the [PersianQA](https://github.com/sajjjadayobi/PersianQA) dataset.  

The base models were originally fine-tuned on the **PQuAD** dataset by [pedramyazdipoor](https://huggingface.co/pedramyazdipoor):
- [`pedramyazdipoor/persian_xlm_roberta_large`](https://huggingface.co/pedramyazdipoor/persian_xlm_roberta_large)
- [`pedramyazdipoor/parsbert_question_answering_PQuAD`](https://huggingface.co/pedramyazdipoor/parsbert_question_answering_PQuAD)

In this project, both models were further fine-tuned on **PersianQA** to improve performance on Persian machine reading comprehension.


## Project Structure

```
├── persianqa_bert.ipynb # Fine-tuning ParsBERT on PersianQA
├── PersianQA_xlm.ipynb # Fine-tuning XLM-RoBERTa on PersianQA
├── demo.py # Inference script to run QA with trained models
├── demo.ipynb # Inference script to run QA with trained models ( in ipynb)
├── Dataset/ # Contains local copy of PersianQA dataset
└── README.md # Project documentation
```


## Dataset

The [PersianQA](https://github.com/sajjjadayobi/PersianQA) dataset is a large-scale Persian question answering dataset with:
- Thousands of question–answer pairs
- Context passages from Persian Wikipedia and other sources
- Both answerable and unanswerable questions (SQuAD v2 style)

A local copy of the dataset is provided in the `Dataset/` directory.


## 📈 Results Summary

| Model & Method               | F1 Score | EM Score | No-Answer F1 |
|-----------------------------|----------|----------|--------------|
| ParsBERT (LoRA + QA Head)   | 69.9     | 56.8     | 86.4         |
| XLM-R ( LoRA + QA Head)    | **85.3** | **71.6** | **90.7**     |


## Models

### 1. Fine-tuned XLM-RoBERTa
Hugging Face Hub: [`mmatinm/mpersian_xlm_roberta_large`](https://huggingface.co/mmatinm/mpersian_xlm_roberta_large)  

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering

repo_id = "mmatinm/mpersian_xlm_roberta_large"

# Load the tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForQuestionAnswering.from_pretrained(repo_id)
```


### 2. LoRA Fine-tuned ParsBERT

Hugging Face Hub: [`mmatinm/parsbert_question_answering_PersianQA_m`]((https://huggingface.co/mmatinm/parsbert_question_answering_PersianQA_m)

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel, PeftConfig

model_path = "mmatinm/parsbert_question_answering_PersianQA_m"

peft_config = PeftConfig.from_pretrained(model_path)
base_model = AutoModelForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path)
model = PeftModel.from_pretrained(base_model, model_path)
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
```

## How to Run Inference

You can test both models using the provided `demo.py` script:

```bash
python demo.py
```

Example output:
```
Context: کوروش بزرگ بنیان‌گذار هخامنشیان بود و در حدود ۲۵۰۰ سال پیش در ایران حکومت می‌کرد.
Question: بنیان‌گذار هخامنشیان چه کسی بود؟

Bert Model Answer: کوروش بزرگ
XLM-Roberta Model Answer: کوروش بزرگ
```





