import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')

def gemini_request(temperature_value, duration_value, text_content, prompt_content):
    genai.configure(api_key=GOOGLE_API_KEY)
    
    print("Temperature Value:", temperature_value)
    print("Duration Value:", duration_value)
    print("Prompt Content:", prompt_content)
    print("Text Content:", text_content)

    generation_config = genai.GenerationConfig(
        temperature=temperature_value,
        top_p=1,
        top_k=0,
        max_output_tokens=8192,
    )
    
    safety_settings = [
        {
            "category": "HARM_CATEGORY_HARASSMENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_HATE_SPEECH",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
        {
            "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
            "threshold": "BLOCK_MEDIUM_AND_ABOVE"
        },
    ]
    
    model = genai.GenerativeModel(
        model_name="gemini-1.5-flash",
        generation_config=generation_config,
        safety_settings=safety_settings
    )
    
    convo = model.start_chat()
    
    try:
        message =  prompt_content + text_content
        convo.send_message(message)
        print(convo.last.text)
        return {"status": "success", "output": convo.last.text}
    except:
        print("429: Resource exhausted")
        return {"status": "failure", "output": None}


