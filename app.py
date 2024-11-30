import requests
import os
from PyPDF2 import PdfReader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
from datasets import Dataset
import json
import pdfplumber
# Step 1: Automate PDF Data Extraction
# Function to download PDFs from a given URL
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded PDF: {save_path}")

# Function to extract text from PDF
import pdfplumber

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text
import pdfplumber

# Function to extract text from PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text



# Example URLs of PDFs 
pdf_urls = ["https://content.dgft.gov.in/Website/CIEP.pdf", "https://content.dgft.gov.in/Website/TF.pdf","https://content.dgft.gov.in/Website/EI.pdf","https://content.dgft.gov.in/Website/GAE.pdf"]
pdf_folder = "./pdfs"
os.makedirs(pdf_folder, exist_ok=True)

# Download and extract text from each PDF
pdf_texts = []
for url in pdf_urls:
    pdf_name = os.path.join(pdf_folder, os.path.basename(url))
    download_pdf(url, pdf_name)
    text = extract_text_from_pdf(pdf_name)
    pdf_texts.append(text)

# Step 2: Fine-Tuning the GPT-2 Model on Extracted Data
# Prepare dataset for fine-tuning (using the extracted PDF text)
dataset = []
for text in pdf_texts:
    # Generate a sample question for the content
    question = f" What is the main idea of this text? "
    dataset.append({"prompt": text, "completion": question})

# Save dataset in JSONL format for fine-tuning
jsonl_file = "fine_tuning_dataset.jsonl"
with open(jsonl_file, "w") as f:
    for entry in dataset:
        f.write(json.dumps(entry) + "\n")

# Step 3: Load Pre-trained GPT-2 Model and Tokenizer
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})
model.resize_token_embeddings(len(tokenizer))
tokenizer.pad_token = tokenizer.eos_token 
# Tokenize the dataset
def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples['prompt'], truncation=True, padding="max_length", max_length=512)
    # Add labels (same as input_ids for causal language modeling)
    tokenized_inputs["labels"] = tokenized_inputs["input_ids"].copy()  
    return tokenized_inputs

# Create a Hugging Face dataset from the JSONL file
dataset = Dataset.from_json(jsonl_file)

# Apply tokenization to the dataset
tokenized_dataset = dataset.map(tokenize_function, batched=True)

# Step 4: Fine-Tune the GPT-2 Model
training_args = TrainingArguments(
    output_dir="./results",  
    eval_strategy="no",
    learning_rate=2e-5,
    per_device_train_batch_size=2,
    num_train_epochs=3,
    save_steps=10_000,
    save_total_limit=2,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    #eval_dataset=eval_dataset 
)

# Fine-tune the model
trainer.train()

# Save the fine-tuned model
trainer.save_model("./fine_tuned_gpt2")

# Step 5: Generate Questions Using the Fine-Tuned Model
# Load the fine-tuned model
fine_tuned_model = GPT2LMHeadModel.from_pretrained("./fine_tuned_gpt2")
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained("./fine_tuned_gpt2")

# Function to generate questions from extracted text
def generate_question(text):
    input_ids = fine_tuned_tokenizer.encode(text, return_tensors="pt")
    
    output = fine_tuned_model.generate(input_ids, max_new_tokens=100, num_return_sequences=1)  
    question = fine_tuned_tokenizer.decode(output[0], skip_special_tokens=True)
    return question

# Test question generation from a sample PDF content
sample_text = pdf_texts[0]  # Using the first PDF's extracted text
generated_question = generate_question(sample_text)
print(f"Generated Question: {generated_question}")
