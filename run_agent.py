"""
End-to-end driver for the local AI patent agent.

Usage:
    python run_agent.py "A magnetic water bottle holder for bicycles that locks automatically"
    python run_agent.py --file idea.txt
    python run_agent.py --skip-retrieval "your idea"       # offline mode (mock candidates)
    python run_agent.py --model qwen2.5:1.5b "your idea"
"""
from __future__ import annotations

import os

# Silence the harmless HuggingFace symlink warning on Windows.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from input_analysis import (
    PatentQueryAnalysis,
    analyze_and_classify_idea,
    build_clients,
)
from prior_evidence_retrieval import AsyncPatentRetriever
from cross_encoder_reranker import PatentReranker
from langgraph_state_machine import patent_synthesis_pipeline


# ------------------------------------------------------------------
# Step 1 → Step 2 bridge
# ------------------------------------------------------------------
def extract_cpc_codes(chroma_results: Dict[str, Any], top_n: int = 2) -> List[str]:
    """Trim full CPC codes (B62J11/00) down to subclasses (B62J)."""
    ids = chroma_results["ids"][0][:top_n]
    subclasses: List[str] = []
    for full in ids:
        m = full.split("/")[0]      # B62J11
        subclass = m[:4]            # B62J
        if subclass and subclass not in subclasses:
            subclasses.append(subclass)
    return subclasses


# ------------------------------------------------------------------
# Step 2 → Step 3 bridge
# ------------------------------------------------------------------
def docs_to_candidates(docs) -> List[Dict[str, Any]]:
    """Convert PriorArtDocument list into reranker candidate dicts."""
    out = []
    for d in docs:
        text = getattr(d, "claims_text", "") or ""
        if not text:
            # Fall back to a claim-shaped wrapper around the abstract.
            text = f"1. {d.abstract}"
        out.append({"id": d.doc_id, "title": d.title, "text": text})
    return out


# ------------------------------------------------------------------
# Offline mock candidates (claim-shaped so the reranker can parse them)
# ------------------------------------------------------------------
MOCK_CANDIDATES: List[Dict[str, Any]] = [
    {
        "id": "KR102370398B1",
        "title": "Closure device for attaching container to bicycle",
        "text": (
            "1. A closure device for attaching a container to a bicycle, comprising: "
            "a receiving latch configured to mechanically lock a bottle upon magnetic alignment; "
            "a mounting base attachable to a bicycle frame; "
            "a receptacle defining a cavity sized to receive the bottle."
        ),
    },
    {
        "id": "US20210169205A1",
        "title": "Device Having Beverage Container Holder",
        "text": (
            "1. A device having a beverage container holder, comprising: "
            "a receiving device for detachably fixing the beverage container in place "
            "using a mechanical release; "
            "a base coupled to a vehicle."
        ),
    },
]


# ------------------------------------------------------------------
# Main pipeline
# ------------------------------------------------------------------
async def run(idea: str, model: str, skip_retrieval: bool, top_k: int) -> Dict[str, Any]:
    print("=" * 70)
    print("STEP 1 — Input analysis + CPC classification")
    print("=" * 70)

    chroma_results = analyze_and_classify_idea(idea, model=model)
    if chroma_results is None:
        print("[driver] Step 1 failed; aborting.")
        sys.exit(1)
    cpc_codes = extract_cpc_codes(chroma_results, top_n=2)
    print(f"[driver] CPC subclasses passed to Step 2: {cpc_codes}")

    # Rebuild the structured analysis (analyze_and_classify_idea only returns Chroma results).
    client, _, _ = build_clients()
    analysis: PatentQueryAnalysis = client.chat.completions.create(
        model=model,
        response_model=PatentQueryAnalysis,
        messages=[
            {
                "role": "system",
                "content": "Extract patent search features. Return only valid JSON.",
            },
            {"role": "user", "content": idea},
        ],
        temperature=0.0,
    )

    print("\n" + "=" * 70)
    print("STEP 2 — Multi-query prior-art retrieval")
    print("=" * 70)

    if skip_retrieval:
        print("[driver] --skip-retrieval set, using mock candidates.")
        candidates = list(MOCK_CANDIDATES)
    else:
        retriever = AsyncPatentRetriever()
        try:
            docs = await retriever.execute_parallel_retrieval(analysis, cpc_codes)
        except Exception as exc:
            print(f"[driver] retrieval failed ({exc}); falling back to mocks.")
            docs = []
        if not docs:
            print("[driver] no live results; falling back to mock candidates.")
            candidates = list(MOCK_CANDIDATES)
        else:
            candidates = docs_to_candidates(docs)

    print(f"[driver] {len(candidates)} candidates going into reranker.")

    print("\n" + "=" * 70)
    print("STEP 3 — Independent-claim parsing + cross-encoder reranking")
    print("=" * 70)

    reranker = PatentReranker(model_name="BAAI/bge-reranker-base")
    ranked = reranker.max_clause_rerank(idea, candidates, top_k=top_k)

    print(f"\n[driver] Top {len(ranked)} reranked patents:")
    for r in ranked:
        snippet = r["matched_limitation"][:90].replace("\n", " ")
        print(f"  {r['patent_id']:<20} score={r['max_similarity_score']:.4f}  | {snippet}...")

    if not ranked:
        print("[driver] No candidates survived reranking; nothing to analyze.")
        return {"final_report": {"novelty_status": "Potentially Novel",
                                 "uncovered_gaps": [],
                                 "suggested_modifications": [],
                                 "legal_disclaimer": "No prior art retrieved."}}

    print("\n" + "=" * 70)
    print("STEP 4 — LangGraph legal reasoning")
    print("=" * 70)

    initial_state = {
        "user_idea": idea,
        "reranked_patents": ranked,
        "deconstructed_idea": [],
        "deconstructed_patents": [],
        "diff_matrix": {},
        "obviousness_analysis": {},
        "final_report": {},
    }
    final_state = patent_synthesis_pipeline.invoke(initial_state)
    return final_state


def main():
    ap = argparse.ArgumentParser(description="Local AI patent agent")
    ap.add_argument("idea", nargs="?", help="Invention description (quoted string)")
    ap.add_argument("--file", help="Read idea from a text file instead")
    ap.add_argument("--model", default="llama3.2:3b", help="Ollama model name")
    ap.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Skip Step 2 and use built-in mock candidates (offline mode)",
    )
    ap.add_argument("--top-k", type=int, default=5, help="How many patents to keep after reranking")
    args = ap.parse_args()

    if args.file:
        idea = Path(args.file).read_text(encoding="utf-8").strip()
    elif args.idea:
        idea = args.idea
    else:
        ap.error("Provide an idea string or --file path")

    final_state = asyncio.run(run(idea, args.model, args.skip_retrieval, args.top_k))

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(json.dumps(final_state.get("final_report", {}), indent=2))


if __name__ == "__main__":
    main()