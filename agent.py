import warnings
import logging
import sys
import os
import contextlib
import io

from fastapi import FastAPI
from pydantic import BaseModel  

# Define the request body structure
class Query(BaseModel):
    question: str

# Create the FastAPI app instance
app = FastAPI()

# === Suppress warnings and logs to keep output clean ===
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
logging.getLogger().setLevel(logging.CRITICAL)

# === Import AI components ===
from transformers import pipeline
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from langchain_huggingface import HuggingFaceEmbeddings  

from langchain_community.llms import OpenAI  

# === Configure AI Settings ===
import torch

device = torch.device("cpu")  # Force CPU usage
# Settings.llm = None  # Disable OpenAI or other external LLMs
from llama_index.llms.openai import OpenAI

import os
os.environ["OPENAI_API_KEY"] = "Your_API_key"

Settings.llm = OpenAI(model="gpt-3.5-turbo")  # or your preferred model
Settings.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load the QA pipeline (HuggingFace model)
qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

# Load documents from the "data/" folder
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)

# Suppress LlamaIndex's internal print about MockLLM
with contextlib.redirect_stdout(io.StringIO()):
    query_engine = index.as_query_engine()

# === FastAPI Route ===
@app.post("/ask")
async def ask(query: Query):
    """Receives a question, runs the retrieval + QA pipeline, and returns the answer."""
    retrieved = query_engine.query(query.question)
    retrieved_context = retrieved.response

    hf_answer = qa_pipeline({
        "context": retrieved_context,
        "question": query.question
    })

    return {"answer": hf_answer["answer"]}