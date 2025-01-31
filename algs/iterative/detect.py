from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

from enums import *

roberta_tokenizer = AutoTokenizer.from_pretrained("openai-community/roberta-base-openai-detector")
roberta_model = AutoModelForSequenceClassification.from_pretrained("openai-community/roberta-base-openai-detector")

def detectAI(text: str, method: Detect) -> float:
    """
    Detects AI given the text and the selected method.

    :param text: The text to detect.
    :param method: The detection method to use.
    :return: The detection score.
    """

    if method == Detect.ZEROGPT:
        return 0.1
    elif method == Detect.GPTZERO:
        return 0.2
    elif method == Detect.ROBERTA:
        return roberta_detect(text)
    else:
        raise ValueError("Invalid method")

def roberta_detect(text: str) -> float:
    inputs = roberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        outputs = roberta_model(**inputs)

    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=-1)
    prob_gpt_generated = 1 - probabilities[0][1].item()
    
    return prob_gpt_generated
