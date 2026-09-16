import json
from typing import List

import chromadb
import instructor
import torch
from chromadb.utils import embedding_functions
from openai import OpenAI
from pydantic import BaseModel, Field


# ==========================================
# 1. PYDANTIC SCHEMA
# ==========================================
class PatentQueryAnalysis(BaseModel):
    summary: str = Field(..., description="Brief single-sentence technical summary of the core invention.")
    technical_mechanisms: List[str] = Field(..., description="Key mechanical, electrical, or software mechanisms used.")
    legal_synonyms: List[str] = Field(..., description="Broader patent-prose legal terms.")
    search_query: str = Field(..., description="Dense space-separated search string.")


# ==========================================
# 2. CLIENTS / DB (built lazily)
# ==========================================
def build_clients():
    ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
    client = instructor.from_openai(ollama_client, mode=instructor.Mode.JSON)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2",
        device=device,
    )
    chroma_client = chromadb.PersistentClient(path="./cpc_db")
    try:
        collection = chroma_client.get_collection(name="cpc_codes", embedding_function=embedding_fn)
    except Exception:
        collection = chroma_client.create_collection(name="cpc_codes", embedding_function=embedding_fn)
    return client, collection, device


SAMPLE_CPC_DATA = [
    {"id": "B62J11/00", "description": "Arrangements supporting or attaching articles on cycle frames or handlebars."},
    {"id": "B65D47/20", "description": "Closures with discharging devices or valves for liquid containers."},
    {"id": "H01F7/02", "description": "Permanent magnets; Magnets or magnetic devices for holding or latching objects."},
    {"id": "A45F5/00", "description": "Holders or carriers for portable articles worn or carried on the body."},
    {"id": "B60R11/00", "description": "Arrangements for keeping or holding articles on or in vehicles."},
]


def ensure_seeded(collection) -> None:
    if collection.count() == 0:
        collection.add(
            ids=[i["id"] for i in SAMPLE_CPC_DATA],
            documents=[i["description"] for i in SAMPLE_CPC_DATA],
            metadatas=[{"cpc_code": i["id"]} for i in SAMPLE_CPC_DATA],
        )
        print("✓ Seeded ChromaDB with sample CPC data.")


# ==========================================
# 3. PIPELINE
# ==========================================
def analyze_and_classify_idea(user_idea: str, model: str = "llama3.2:3b"):
    client, collection, device = build_clients()
    ensure_seeded(collection)

    print(f"\n[Raw Input Idea]: {user_idea}\n")

    system_instruction = (
        "You are a Patent Classification Expert. Extract key technical mechanisms and legal synonyms "
        "from the user's invention description to form a focused patent search query. "
        "Return only valid JSON matching the schema."
    )

    response: PatentQueryAnalysis = client.chat.completions.create(
        model=model,
        response_model=PatentQueryAnalysis,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_idea},
        ],
        temperature=0.0,
    )

    print("--- Extracted Structured Features ---")
    print(json.dumps(response.model_dump(), indent=2))

    try:
        results = collection.query(query_texts=[response.search_query], n_results=3)
    except Exception as exc:
        print(f"[ChromaDB query failed]: {exc}")
        return None

    print("\n--- Top Matching CPC Classes ---")
    for i in range(len(results["ids"][0])):
        print(
            f"Rank {i+1} | Code: {results['ids'][0][i]} "
            f"| Distance: {results['distances'][0][i]:.4f}\n"
            f"  Description: {results['documents'][0][i]}\n"
        )
    return results


if __name__ == "__main__":
    analyze_and_classify_idea(
        "A magnetic water bottle holder for bicycles that locks automatically"
    )