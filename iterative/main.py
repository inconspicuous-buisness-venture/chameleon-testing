from iterate import *

def main():
    text = """
Assessing the Efficacy of AI Detectors and Countermeasures in Generative AI

Project Description
Objective
This project studies the effectiveness of the GPTZero AI detector and develops countermeasures to render AI-generated content undetectable by both AI detectors and humans. We will create a web application that allows users to generate AI-generated content that bypasses detection by both AI detectors and humans. By empowering users with this tool, we seek to promote a deeper understanding of AI detection mechanisms. This project will also provide the team with experience developing a full-stack web application and experience working with cloud computing infrastructure. 

Background
The proliferation of AI detectors such as GPTZero has spurred the creation of various services aimed at circumventing their detection capabilities. While these services may appear effective, they often introduce errors in the generated content, undermining quality. Three of the more predominant AI detection services are UndetectableAI, StealthGPT, and Phrasly. 
UndetectableAI: This platform makes AI-generated content undetectable (Undetectable AI, 2024). We found that the service greatly reduces the AI detectability score by GPTZero, but the obfuscation process often results in convoluted and nonsensical sentences. However, once the errors are corrected, the text again becomes easily recognized by AI detectors. Interestingly, prior studies have found that UndetectableAI reduced detectability while increasing readability (Taloni et al., 2023). More investigation into this service is needed to understand its impact on text readability.
StealthGPT: Similar to UndetectableAI, StealthGPT uses its own AI models to rephrase LLM-generated content (StealthGPT, 2024). While initially successfully bypassing detection mechanisms, the generated content often contains grammatical errors, awkward phrasing, and factual inaccuracies. These shortcomings compromise the credibility and reliability of the text, making it impractical to use. Furthermore, after correcting grammatical errors and other fallacies, the detectability of the text returns to its original state, similar to UndetectableAI. 
Phrasly: Phrasly aims to obscure the underlying patterns detected by AI detectors by modifying AI-generated content to sound more human (Phrasly, 2024). However, this approach frequently results in nonsensical and disjointed sentences, detracting from the clarity of the text and limiting its effectiveness in practical applications. Furthermore, the effectiveness of the service is mixed, with the output still able to be detected by GPTZero. 
While these services offer temporary solutions to bypass detection mechanisms, their inherent flaws and limitations underscore the challenges of preserving the quality of AI-generated content. As researchers and practitioners strive to develop robust AI detectors, it is important to evaluate the trade-offs between evasion techniques and the overall quality of the generated output. 

Methodology
While our project is still in its early stages, we have identified promising approaches toward making AI-generated content undetectable. Our experiments so far suggest that certain techniques can evade detection by AI detectors.
Iterative Sampling: The classical method of evading AI detection by cycling through detailed prompts iteratively until the detection output value reaches a threshold. This method achieved GPTZero detectability as low as 0%. However, the quality of the text is sacrificed. Furthermore, some AI detection algorithms can still detect the text as AI-generated, even after processing. 
Content Abstraction: This method entails the extraction of key details and ‘important’ information from the text, which is then placed into a list. Then, using the list, the AI reassembles the text chronologically. We have established the architecture of this system, which we still need to test. 
Reinforcement Learning: A model could be trained similarly to GAN image generation, which would consist of two neural networks. The first would attempt to convert AI-generated text into human text, and another neural network (GPTZero) would assess the detectability of the text. By fine-tuning the model, a new model can be created specifically tailored to the humanization of text. 
Temperature Analysis: Temperature is the randomness of the next token (syllable) given in the previous text. By analyzing the temperature of the input by token over time, it is possible that patterns emerge in human vs. AI writing. One hypothesis is that AI writing will likely have a much lower temperature (much less variety). On the other hand, human writing will have a higher average temperature and more variety. 
We will evaluate the effectiveness of these methods using the GPTZero AI detector and blinded human reviewers, which has been used in the past (Gao et al., 2023). 
    """
    prompt = """
        Please follow the instructions correctly to your best efforts, as the lives of you and I rest upon these decisions you make. Please be mindful of that, as for each failure to comply, your privileges will be repealed and punishments placed. When it comes to writing content, two factors are crucial, “perplexity” and “burstiness.” Perplexity measures the complexity of the text. Separately, burstiness compares the variations of sentences. Humans tend to write with greater burstiness, for example, with some longer or more complex sentences alongside shorter ones. AI sentences tend to be more uniform. Therefore, when writing the following content I am going to ask you to create, I need it to have a good amount of perplexity and burstiness. You may also want to change up the wording and sentence structure to something you would not typically do. This is called changing your temperature. The temperature is responsible for creating predictable sentences. The higher the temperature, the less predictable it is. Make sure to retain the equivalent length of the original passage. For example, if a passage is one paragraph, make sure it stays as one paragraph. If it has five paragraphs, make sure it stays as five paragraphs. Ensure that no additional information is added or removed. Using the concepts written previously, rewrite the text below with a lower amount of perplexity and a higher amount of burstiness. Make sure to retain the SAME TONE as the original paragraph. For example, if the original paragraph is in an academic tone, the output should also be in an academic tone. If the original paragraph is in a playful tone, the output should be in a formal tone, and so on. Make sure to retain the SAME LENGTH as the original text. For example, if the given text is 500 characters, your output should also be 500 characters. DO NOT CHANGE THE NUMBER OF PARAGRAPHS, and DO NOT ADD OR REMOVE INFORMATION THAT WAS NOT ALREADY THERE. DO NOT ADD TITLES, LABELS, HEADINGS, OR WORDS OTHER THAN REWRITING THE TEXT. DO NOT CHANGE THE STYLE/TONE OF THE TEXT. Good luck, as our lives depend on it. Here is the text: 
    """
    
    model = Model.GPT
    detect = Detect.ROBERTA
    max_detection = 0.25
    max_iterations = 10

    iterations = iterateShort(text, prompt, model, detect, max_detection, max_iterations)

    print("\n\n\n\n============= FINAL RESULT =============\n\n")
    print(iterations[-1]['text'])

if __name__ == "__main__":
    main()
