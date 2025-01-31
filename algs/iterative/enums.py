from enum import Enum

class Model(Enum):
    GEMINI = "Gemini 1.0 Pro"
    GPT = "GPT-4o"
    CLAUDE = "Claude 3 Sonnet"

class Detect(Enum):
    ZEROGPT = "ZeroGPT"
    GPTZERO = "GPTZero"
    ROBERTA = "RoBERTa"
