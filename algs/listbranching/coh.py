import torch
from transformers import RobertaForSequenceClassification, RobertaTokenizer

# Define the path to the model
model_path = r"C:\Users\Makai\Downloads\best_roberta_model.pt"

# Load the tokenizer
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

# Load the model architecture (binary classification)
model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

# Load the trained weights
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
model.eval()  # Set model to evaluation mode

# Function to predict
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
    prediction = torch.argmax(logits, dim=1).item()
    print(f"Logits: {logits}")
    print(f"Probabilities: {probabilities}")
    return "Class 1" if prediction == 1 else "Class 0"

# Example of coherent text
text = "This is a sample text for classification."
print(f"Prediction: {predict(text)}")

# Example of incoherent text
incoherent_text = "asdf qwer zxcv poiuy POOP POOP POOP lkjhg mnbv"
print(f"Prediction for incoherent text: {predict(incoherent_text)}")