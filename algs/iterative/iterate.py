from typing import List, Union, Dict
from enums import *
from models import *
from detect import *


def iterateShort(text: str, prompt: str, model: Model, detect: Detect, max_detection: float, max_iterations: int = 15) -> List[Dict[str, Union[str, float]]]:
    """
    Iterates through the samples and rewrites the text until the detection score is less than the maximum detection score or the maximum number of iterations is reached.

    :param text: The text to iterate.
    :param prompt: The prompt to generate text.
    :param model: The model to use.
    :param detect: The detection method to use.
    :param max_detection: The maximum detection score.
    :param max_iterations: The maximum number of iterations.
    :return: The list of iterations as dictionaries.
    """
    
    detection_score = detectAI(text, detect)
    iterations = [{"text": text, "detection_score": detection_score}]
    iteration_count = 0

    while detection_score > max_detection and iteration_count < max_iterations:
        new_text = generateText(text, prompt, model)
        detection_score = detectAI(new_text, detect)
        iterations.append({"text": new_text, "detection_score": detection_score})
        print(f"SCORE: {detection_score} | TEXT: {new_text}\n")

        iteration_count += 1
    
    return iterations








# TODO FINISH THE FUNCTION BELOW
# i literally havent even finished it lol

def iterateLong(text: str, prompt: str, model: Model, detect: Detect, max_detection: float, max_iterations: int = 10) -> List[Dict[str, Union[str, float]]]:
    """
    Iterates through the samples and rewrites the text until the detection score is less than the maximum detection score or the maximum number of iterations is reached. However, it also individually humanizes each paragraph and then measures the detectability of the full text. 

    :param text: The text to iterate.
    :param prompt: The prompt to generate text.
    :param model: The model to use.
    :param detect: The detection method to use.
    :param max_detection: The maximum detection score.
    :param max_iterations: The maximum number of iterations.
    :return: The list of iterations as dictionaries.
    """
    
    iterations = [{"text": text, "detection_score": detectAI(text, detect)}]
    split_text = text.split('\n')

    for paragraph in split_text:
        iteration_count = 0

        while detection_score > max_detection and iteration_count < max_iterations:
            if paragraph.split(' ').len() < 15:
                break;
            
            new_text = generateText(paragraph, prompt, model)
            detection_score = detectAI(new_text, detect)
            split_text[split_text.index(paragraph)] = new_text
            print(f"SCORE: {detection_score} | TEXT: {new_text}\n")

            iteration_count += 1
        
        combined_text = "\n".join(split_text)
        detection_score = detectAI(combined_text, detect)
        iterations.append({"text": "\n".join(combined_text), "detection_score": detection_score})
    
    return iterations
