import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_community.document_loaders import TextLoader  # Supports text/PDF loading

# ===========================
# 1. Load Environment Variables
# ===========================
load_dotenv()
api_key = os.getenv("API_KEY")

if not api_key:
    raise ValueError("API_KEY not found. Please check your .env file.")

# ===========================
# 2. Initialize Language Model
# ===========================
model = "deepseek-r1-distill-llama-70b"
deepseek = ChatGroq(api_key=api_key, model_name=model)

# ===========================
# 3. Setup Output Parsing
# ===========================
parser = StrOutputParser()
deepseek_chain = deepseek | parser  # Chain model with parser for clean output

# ===========================
# 4. Quick Test: Verify Model Interaction
# ===========================
test_response = deepseek_chain.invoke("Hello, there!")
print(f"Model Test Response: {test_response}\n")

# ===========================
# 5. Load External Context Data (Max=6000)
# ===========================
try:
    loader = TextLoader('data.txt', encoding='utf-8')  
    data = loader.load()
except FileNotFoundError:
    raise FileNotFoundError("Error: 'data.txt' not found. Please ensure the file exists.")

# ===========================
# 6. Define Chatbot Prompt Template
# ===========================
template = """ 
You are an AI assistant. Respond using only the given context.  
Do NOT generate information beyond what is provided.  

Context:  
{context}  

Question:  
{question}  
"""

# ===========================
# 7. Generate AI Response
# ===========================
question = "What is the defunct nature of cybersecurity solutions, with regards to incidences?"
formatted_prompt = template.format(context=data, question=question)
answer = deepseek_chain.invoke(formatted_prompt)

# ===========================
# 8. Output Final Response
# ===========================
print("Chatbot Response:\n", answer)