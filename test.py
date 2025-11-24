from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoConfig
import torch

# ✅ Path to your saved model
path = r"F:\Final_Year_Project\legal-ai\models\t5_base_summarizer"

# ✅ Step 1: Load base config (ensures d_model=768)
config = AutoConfig.from_pretrained("t5-base")

# ✅ Step 2: Load tokenizer from same base
tokenizer = T5Tokenizer.from_pretrained("t5-base")

# ✅ Step 3: Load fine-tuned model with same architecture
model = T5ForConditionalGeneration.from_pretrained(path, config=config)

print("✅ Model loaded successfully!")
print("Hidden size:", model.config.d_model)   # should be 768

# ✅ Step 4: Quick sanity check
text = "The vendor failed to meet the data protection requirements."
input_ids = tokenizer("summarize: " + text, return_tensors="pt").input_ids
output_ids = model.generate(input_ids, max_length=40)
print("📝 Summary:", tokenizer.decode(output_ids[0], skip_special_tokens=True))
