import json
import chromadb
from chromadb.utils import embedding_functions
import instructor
from openai import OpenAI
from pydantic import BaseModel, Field
import torch

# load the model and test with sample input


# ==========================================
# 1. SETUP LOCAL OLLAMA CLIENT WITH INSTRUCTOR
# ==========================================
ollama_client = OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama", # OpenAI client requires a non-empty string
)

# Use JSON mode for local Ollama endpoints
client = instructor.from_openai(ollama_client, mode=instructor.Mode.JSON)


# ==========================================
# 2. DEFINE PYDANTIC SCHEMA
# ==========================================
class PatentQueryAnalysis(BaseModel):
    summary: str = Field(
        ...,
        description="Brief single-sentence technical summary of the core invention.",
    )
    technical_mechanisms: list[str] = Field(
        ...,
        description="Key mechanical, electrical, or software mechanisms used.",
    )
    legal_synonyms: list[str] = Field(
        ...,
        description="Broader patent-prose legal terms (e.g., 'bicycle' -> 'cyclical vehicle').",
    )
    search_query: str = Field(
        ...,
        description="A dense space-separated search string combining primary mechanisms and legal synonyms.",
    )


# ==========================================
# 3. INITIALIZE CHROMADB VECTOR DATABASE
# ==========================================
device = "cuda" if torch.cuda.is_available() else "cpu"
embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2",
    device=device
)

chroma_client = chromadb.PersistentClient(path="./cpc_db")

# Safer collection retrieval pattern
try:
    collection = chroma_client.get_collection(
        name="cpc_codes", 
        embedding_function=embedding_fn
    )
except Exception:
    collection = chroma_client.create_collection(
        name="cpc_codes", 
        embedding_function=embedding_fn
    )

sample_cpc_data = [
    {
        "id": "B62J11/00",
        "description": "Arrangements supporting or attaching articles on cycle frames or handlebars.",
    },
    {
        "id": "B65D47/20",
        "description": "Closures with discharging devices or valves for liquid containers.",
    },
    {
        "id": "H01F7/02",
        "description": "Permanent magnets; Magnets or magnetic devices for holding or latching objects.",
    },
    {
        "id": "A45F5/00",
        "description": "Holders or carriers for portable articles worn or carried on the body.",
    },
    {
        "id": "B60R11/00",
        "description": "Arrangements for keeping or holding articles on or in vehicles.",
    },
]

if collection.count() == 0:
    collection.add(
        ids=[item["id"] for item in sample_cpc_data],
        documents=[item["description"] for item in sample_cpc_data],
        metadatas=[{"cpc_code": item["id"]} for item in sample_cpc_data],
    )
    print(f"✓ Initialized ChromaDB on {device.upper()} and seeded CPC data.")


# ==========================================
# 4. EXECUTION PIPELINE
# ==========================================
def analyze_and_classify_idea(user_idea: str):
    print(f"\n[Raw Input Idea]: {user_idea}\n")

    system_instruction = (
        "You are a Patent Classification Expert. Extract key technical mechanisms and legal synonyms "
        "from the user's invention description to form a focused patent search query."
    )

    # Calling Instructor with Llama 3.2
    response: PatentQueryAnalysis = client.chat.completions.create(
        model="llama3.2:3b",
        response_model=PatentQueryAnalysis,
        messages=[
            {"role": "system", "content": system_instruction},
            {"role": "user", "content": user_idea},
        ],
        temperature=0.0,  # Strict zero temperature for maximum extraction determinism
    )

    print("--- Extracted Structured Features ---")
    print(json.dumps(response.model_dump(), indent=2))

    # Vector search against local ChromaDB
    results = collection.query(
        query_texts=[response.search_query], n_results=3
    )

    print("\n--- Top Matching CPC Classes ---")
    for i in range(len(results["ids"][0])):
        cpc_code = results["ids"][0][i]
        description = results["documents"][0][i]
        distance = results["distances"][0][i]
        print(
            f"Rank {i+1} | Code: {cpc_code} | Distance: {distance:.4f}\n  Description: {description}\n"
        )


if __name__ == "__main__":
    raw_idea = (
        "A magnetic water bottle holder for bicycles that locks automatically"
    )
    analyze_and_classify_idea(raw_idea)