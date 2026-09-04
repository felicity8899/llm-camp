import json
import os
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

# Load environment variables from the .env file in the project root directory
load_dotenv()

def ingest_cfa_data(json_path, persist_directory="./chroma_db"):
    # 1. Read the JSON data
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    documents = []
    
    # 2. Convert JSON objects into LangChain Document format
    for item in data:
        content = f"Module: {item.get('module', '')}\n"
        content += f"Question: {item['question_text']}\n"
        content += f"Correct Answer: {item['correct_answer']}\n"
        content += f"Explanation: {item['explanation']}"
        
        metadata = {
            "question_id": item.get("question_id", ""),
            "module": item.get("module", "")
        }
        
        doc = Document(page_content=content, metadata=metadata)
        documents.append(doc)

    print(f"Successfully loaded {len(documents)} CFA knowledge chunks.")

    # 3. Initialize the cloud-based Embedding model (automatically reads OPENAI_API_KEY from os.environ)
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # 4. Create and persist the vector database
    print("Generating embeddings and storing them in the Chroma database. Please wait...")
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=persist_directory
    )
    
    print(f"RAG knowledge base built successfully! Vector data is saved in the {persist_directory} directory.")

if __name__ == "__main__":
    ingest_cfa_data("data/cfa_questions.json")