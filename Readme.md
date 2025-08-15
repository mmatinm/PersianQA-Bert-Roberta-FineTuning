# Persian Question Answering with Fine-Tuned Transformers

This repository contains the code and resources for fine-tuning state-of-the-art transformer models (BERT and XLM-RoBERTa) on the PersianQA dataset for extractive question answering in Persian. The goal is to provide highly accurate models capable of answering questions based on provided text contexts in Farsi.

---

## Dataset

The **PersianQA** dataset is a large-scale Persian Question Answering dataset. It is specifically designed for extractive question answering tasks, where the answer to a question is a span of text directly extracted from a given context.

You can find more details and download the dataset from its official GitHub repository:
[https://github.com/sajjjadayobi/PersianQA](https://github.com/sajjjadayobi/PersianQA)

---

## Fine-Tuned Models

Two transformer models have been fine-tuned and are available on Hugging Face for easy integration into your projects:

1.  **`parsbert_question_answering_PersianQA_m` (based on ParsBERT)**
    * **Hugging Face Link:** [mmatinm/parsbert_question_answering_PersianQA_m](https://huggingface.co/mmatinm/parsbert_question_answering_PersianQA_m)
    * This model utilizes a PEFT (Parameter-Efficient Fine-Tuning) approach, specifically LoRA, for efficient deployment and usage.

2.  **`mpersian_xlm_roberta_large` (based on XLM-RoBERTa Large)**
    * **Hugging Face Link:** [mmatinm/mpersian_xlm_roberta_large](https://huggingface.co/mmatinm/mpersian_xlm_roberta_large)
    * This is a full fine-tuned model based on the large multilingual XLM-RoBERTa architecture.

---

## How to Load and Use the Models

You can easily load and use these models for inference using the `transformers` and `peft` libraries.

### Loading `parsbert_question_answering_PersianQA_m` (PEFT/LoRA Model)

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel, PeftConfig
import torch

# Specify the model path on Hugging Face
model_path = "mmatinm/parsbert_question_answering_PersianQA_m"

# Load PEFT configuration
peft_config = PeftConfig.from_pretrained(model_path)

# Load the base model
base_model = AutoModelForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path)

# Load the PEFT model (LoRA adapter) on top of the base model
model = PeftModel.from_pretrained(base_model, model_path)

# Load the tokenizer for the base model
tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)

# Ensure model is in evaluation mode
model.eval()

# Example usage (CPU inference for simplicity, use .to('cuda') for GPU)
context = "کوروش بزرگ بنیان‌گذار هخامنشیان بود و در حدود ۲۵۰۰ سال پیش در ایران حکومت می‌کرد."
question = "بنیان‌گذار هخامنشیان چه کسی بود؟"

inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits) + 1 # +1 to include the end token

answer_tokens = inputs["input_ids"][0][start_index:end_index]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"Context: {context}")
print(f"Question: {question}")
print(f"Answer: {answer}")
```

### Loading `mpersian_xlm_roberta_large` (Full Fine-Tuned Model)

```python
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import torch

# Specify the repository ID on Hugging Face
repo_id = "mmatinm/mpersian_xlm_roberta_large"

# Load the tokenizer and model directly
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model = AutoModelForQuestionAnswering.from_pretrained(repo_id)

# Ensure model is in evaluation mode
model.eval()

# Example usage (CPU inference for simplicity, use .to('cuda') for GPU)
context = "کشور ایران با پایتخت تهران، در غرب آسیا واقع شده است."
question = "پایتخت ایران کجاست؟"

inputs = tokenizer(question, context, return_tensors="pt", truncation=True)

with torch.no_grad():
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits

start_index = torch.argmax(start_logits)
end_index = torch.argmax(end_logits) + 1 # +1 to include the end token

answer_tokens = inputs["input_ids"][0][start_index:end_index]
answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)

print(f"Context: {context}")
print(f"Question: {question}")
print(f"Answer: {answer}")
```

---

## Demo

The `demo.py` script provides a practical example of how to load and use both fine-tuned models for inference. It includes a simple function to clean text, load the models, and get answers to your questions from a given context.

To run the demo, make sure you have the required libraries installed and execute:
```bash
python demo.py
```

---

## Installation

To get started, you'll need to install the necessary Python libraries. It's recommended to create a virtual environment.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

# Install required packages
pip install transformers torch peft scikit-learn
```

---

## Notebooks

* `persianqa_bert.ipynb`: Jupyter notebook for fine-tuning a BERT-based model on the PersianQA dataset.
* `PersianQA_xlm.ipynb`: Jupyter notebook for fine-tuning an XLM-RoBERTa-based model on the PersianQA dataset.
* `demo.py`: Python script demonstrating inference with the fine-tuned models.

---

## Acknowledgements

* **PersianQA Dataset:** We extend our gratitude to Sajjad Ayobi and the creators of the PersianQA dataset for providing this valuable resource for Persian NLP research.

---

## Contact

For any questions or suggestions, feel free to open an issue in this repository or contact the maintainer directly.

