import os
import chromadb
from google import genai
from dotenv import load_dotenv

load_dotenv()

gemini_client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

def embed_text(text):
    result = gemini_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

chroma_client = chromadb.PersistentClient(path="./chroma_db")
collection = chroma_client.get_or_create_collection(name="arxiv_papers")