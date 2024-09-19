import google.generativeai as genai
import os
from dotenv import load_dotenv

from enums import *

load_dotenv()
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY_2')

def generateText(text: str, prompt: str, model: Model) -> str:
    """
    Generates text given the text, prompt, and selected model.

    :param text: The text to generate.
    :param prompt: The prompt to generate text.
    :param model: The model to use.
    :return: The generated text.
    """

    if model == Model.GEMINI:
        return gemini(text, prompt)
    
    elif model == Model.GPT:
        return "Generated text by GPT-4o"
    
    elif model == Model.CLAUDE:
        return "Generated text by Claude 3 Sonnet"
    
    else:
        return "No model detected"

def gemini(text: str, prompt: str) -> str:
    genai.configure(api_key=GOOGLE_API_KEY)

    generation_config = genai.GenerationConfig(temperature=1, top_p=1, top_k=0, max_output_tokens=8192)
    gemini_model = genai.GenerativeModel(model_name="gemini-1.0-pro", generation_config=generation_config)
    convo = gemini_model.start_chat()

    try:
        convo.send_message(text + prompt)
        return convo.last.text
    except:
        print("429: Resource exhausted. Please use a new API Key.")
    