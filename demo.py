import torch
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from peft import PeftModel, PeftConfig

# -------------------------------
# 1. Text cleaning function (same as training)
# -------------------------------
def clean_text(text):
    import re
    text = text.replace('ك', 'ک').replace('ي', 'ی').replace('ۀ', 'ه').replace('ة', 'ه')
    text = text.replace('أ', 'ا').replace('إ', 'ا').replace('ؤ', 'و').replace('ئ', 'ی').replace('آ', 'ا')
    text = text.replace('\u200c', '').replace('\u0640', '').replace('\u200d', '').replace('\u200e', '').replace('\u200f', '')
    text = re.sub(r'\s*=\s*', '=', text)
    text = re.sub(r'\s*\+\s*', '+', text)
    text = re.sub(r'\s*-\s*', '-', text)
    text = re.sub(r'\s*\*\s*', '*', text)
    text = re.sub(r'\s*/\s*', '/', text)
    text = re.sub(r'\s*–\s*', '–', text)
    text = re.sub(r'\s*…', '...', text).replace('...', '…')
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\s+([.,،؛!?])', r'\1', text)
    text = re.sub(r'([.,،؛!?])([^\s])', r'\1 \2', text)
    text = re.sub(r'[A-Z]', lambda m: m.group(0).lower(), text)
    return text.strip()

# -------------------------------
# 2. Function to load either LoRA or normal model
# -------------------------------
def load_model(model_name, is_lora=False):
    if is_lora:
        peft_config = PeftConfig.from_pretrained(model_name)
        base_model = AutoModelForQuestionAnswering.from_pretrained(peft_config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base_model, model_name)
        tokenizer = AutoTokenizer.from_pretrained(peft_config.base_model_name_or_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    return model, tokenizer

# -------------------------------
# 3. Function to run QA on one question/context
# -------------------------------
def answer_question(model, tokenizer, question, context):
    model.eval()
    inputs = tokenizer(
        question,
        context,
        max_length=512,
        truncation="only_second",
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(**{k: v.to(model.device) for k, v in inputs.items()})
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits)

    answer_tokens = inputs["input_ids"][0][start_idx:end_idx+1]
    answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
    return clean_text(answer)

# -------------------------------
# 4. Demo
# -------------------------------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Your context & question
    context = """کوروش بزرگ بنیان‌گذار هخامنشیان بود و در حدود ۲۵۰۰ سال پیش در ایران حکومت می‌کرد."""
    question = """بنیان‌گذار هخامنشیان چه کسی بود؟"""


    context = clean_text(context)
    question = clean_text(question)

    # Load LoRA model
    model1, tokenizer1 = load_model("mmatinm/parsbert_question_answering_PersianQA_m", is_lora=True)
    model1.to(device)
    answer1 = answer_question(model1, tokenizer1, question, context)

    # Load normal model
    model2, tokenizer2 = load_model("mmatinm/mpersian_xlm_roberta_large", is_lora=False)
    model2.to(device)
    answer2 = answer_question(model2, tokenizer2, question, context)

    print("\nContext:", context)
    print("Question:", question)
    print("\nBert Model Answer:", answer1)
    print("XLM-Roberta Model Answer:", answer2)
