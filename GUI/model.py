import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()

GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
PROMPT = os.getenv('PROMPT_A')

# TODO: instead of using a class, these should just be functions. There is no need for a class here.
class ModelAPIs:
    @staticmethod
    def gemini_request(temperature_value, duration_value, text_content, prompt_content):

        genai.configure(api_key=GOOGLE_API_KEY)
        
        print(temperature_value)
        print(duration_value)
        print(prompt_content)
        print(text_content)

        generation_config = {
            "temperature": temperature_value,
            "top_p": 1,
            "top_k": 0,
            "max_output_tokens": 8192,
        }
        
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
            # TODO: pycharm is complaining at the generation_config line:
            # Expected type 'GenerationConfig | GenerationConfigDict | GenerationConfig | None',
            # got 'dict[str, int | Any]' instead
            generation_config=generation_config,
            safety_settings=safety_settings
        )
        
        convo = model.start_chat()
        
        try:
            convo.send_message(PROMPT + text_content)
            print(convo.last.text)
            return {"status": "success", "output": convo.last.text}
        
        except:  # TODO: do not use bare except. List the exception(s) you expect
            print("429: Resource exhausted")
            return {"status": "failure", "output": None}
    