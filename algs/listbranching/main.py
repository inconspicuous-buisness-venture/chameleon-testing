from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B")
model.eval()

def forward_sample(prompt, top_k=4, n_steps=3):
    # Tokenize the input prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    def sample_step(input_ids, depth):
        # Base case: If depth is zero, return the input_ids as a possible sequence
        if depth == 0:
            return [input_ids]

        # Run the model to get logits
        with torch.no_grad():
            outputs = model(input_ids=input_ids)
            logits = outputs.logits[:, -1, :]  # Take the logits of the last token

        # Get the top_k predictions
        probs = torch.softmax(logits, dim=-1)
        top_k_probs, top_k_ids = torch.topk(probs, top_k, dim=-1)

        # Recursively expand each top_k token into further steps
        sequences = []
        for i in range(top_k):
            next_token_id = top_k_ids[0, i].unsqueeze(0).unsqueeze(0)
            next_input_ids = torch.cat([input_ids, next_token_id], dim=-1)
            sequences.extend(sample_step(next_input_ids, depth - 1))
        
        return sequences

    # Start sampling from the given prompt
    sampled_sequences = sample_step(input_ids, n_steps)

    # Decode all sampled sequences to strings
    decoded_sequences = [tokenizer.decode(seq[0], skip_special_tokens=True) for seq in sampled_sequences]
    
    return decoded_sequences

prompt = "Once upon a time"
sampled_texts = forward_sample(prompt, top_k=3, n_steps=4)
for text in sampled_texts:
    print(text)