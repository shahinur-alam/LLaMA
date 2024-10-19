
import os
from transformers import LlamaTokenizer, LlamaForCausalLM
import torch
from huggingface_hub import login


# Disable symlinks warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Set your Hugging Face token as an environment variable
# Replace 'your_actual_token_here' with your real token
os.environ["HUGGINGFACE_TOKEN"] = "HuggingFace Token"

# Login to Hugging Face
login(token=os.environ["HUGGINGFACE_TOKEN"])

# Model name
model_name = "meta-llama/Llama-2-7b-chat-hf"
#model_name ="meta-llama/Llama-3.1-70B"

# Load tokenizer and model
print("Loading tokenizer...")
tokenizer = LlamaTokenizer.from_pretrained(model_name, token=os.environ["HUGGINGFACE_TOKEN"])

print("Loading model...")
model = LlamaForCausalLM.from_pretrained(model_name, token=os.environ["HUGGINGFACE_TOKEN"])

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Model loaded on {device}")

# Input text for testing
input_text = "Who is shahinur?"

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate a response
print("Generating response...")
with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_length=100,
        num_return_sequences=1,
        temperature=0.7,
        top_p=0.95,
    )

# Decode and print the generated text
output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nInput:", input_text)
print("\nOutput:", output_text)