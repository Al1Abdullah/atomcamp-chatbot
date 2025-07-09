# atomcamp Chatbot

atomcamp Chatbot is an AI-powered assistant developed for the atomcamp platform. It is designed to answer queries related to atomcamp's services, programs, and frequently asked questions using a combination of natural language understanding and document-based retrieval.

This chatbot leverages modern NLP techniques including sentence embeddings and vector-based semantic search. The objective is to deliver fast, accurate, and context-aware responses through a simple browser interface.

## Features

- Retrieval-Augmented Generation (RAG) pipeline
- Semantic search powered by FAISS
- Document chunking using RecursiveCharacterTextSplitter
- Embeddings via sentence-transformers/all-MiniLM-L6-v2
- Real-time user interface built with Gradio
- Modular and maintainable Python codebase
- Secure handling of environment variables

## Technologies

- Python 3.10+
- Gradio 5.35.0
- Hugging Face Transformers and Hub
- LangChain
- FAISS
- Sentence Transformers
- dotenv

### 1. Clone the Repository
git clone https://huggingface.co/spaces/ABdullah937e/atomcamp-chatbot
cd atomcamp-chatbot

### 2. Create Virtual Environment
python -m venv venv

### 3. Activate the Virtual Environment

#### 3.1 For Linux/macOS:
source venv/bin/activate

#### 3.2 For Windows:
venv\Scripts\activate

### 4. Install Dependencies
pip install -r requirements.txt

### 5. Set Environment Variable (using .env file)
#### 5.1 Create a file named .env and add the following line:
"env"="28813h28e29e93j"

### 6. Run the App
python app.py
