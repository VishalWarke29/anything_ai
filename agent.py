import warnings
import logging
import sys
import os
import contextlib
import io

from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from transformers import pipeline
from llama_index.core.readers.file.base import SimpleDirectoryReader
from llama_index.core import VectorStoreIndex
from llama_index.core.settings import Settings
from langchain_huggingface import HuggingFaceEmbeddings
from llama_index.llms.openai import OpenAI

import torch

# === Configure AI and Logging Settings ===
warnings.filterwarnings("ignore", category=UserWarning)
logging.getLogger().setLevel(logging.CRITICAL)

device = torch.device("cpu")
os.environ["OPENAI_API_KEY"] = "Your_API_key"  # ← Replace with actual key or load via env var

Settings.llm = OpenAI(model="gpt-3.5-turbo")  # Set LLM
Settings.embed_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2")

app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# === Upload endpoint ===
@app.post("/upload/")
async def upload_pdf(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
    return {"filename": file.filename}

# === Ask question endpoint ===
class Query(BaseModel):
    question: str
    filename: str  # Now includes filename to target the uploaded file

@app.post("/ask/")
async def ask(query: Query):
    file_path = os.path.join(UPLOAD_DIR, query.filename)

    # Load uploaded document dynamically
    documents = SimpleDirectoryReader(input_files=[file_path]).load_data()
    index = VectorStoreIndex.from_documents(documents)

    with contextlib.redirect_stdout(io.StringIO()):
        query_engine = index.as_query_engine()

    retrieved = query_engine.query(query.question)
    retrieved_context = retrieved.response

    hf_answer = qa_pipeline({
        "context": retrieved_context,
        "question": query.question
    })

    return JSONResponse({"answer": hf_answer["answer"]})