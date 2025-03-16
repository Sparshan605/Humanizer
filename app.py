from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch
import streamlit

# Load pre-trained T5 model
model_name = "t5-small"
model = T5ForConditionalGeneration.from_pretrained(model_name)
tokenizer = T5Tokenizer.from_pretrained(model_name)

# Example text
input_text = "paraphrase: AI-generated text that needs humanization."
input_ids = tokenizer.encode(input_text, return_tensors="pt")

# Generate humanized text
output_ids = model.generate(input_ids, max_length=50)
humanized_text =T5Tokenizer.from_pretrained("t5-small", legacy=False)

print(humanized_text)
