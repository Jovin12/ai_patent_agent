import json
import urllib.request as urlrequest
from typing import Any, Dict, List, Literal, TypedDict

import instructor
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field


# ==========================================
# 1. LLM CLIENT
# ==========================================
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
client = instructor.from_openai(ollama_client, mode=instructor.Mode.JSON)


def get_available_ollama_models() -> List[str]:
    try:
        with urlrequest.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            data = json.load(response)
    except Exception:
        return []
    return [
        m.get("name") or m.get("model")
        for m in data.get("models", [])
        if isinstance(m, dict) and (m.get("name") or m.get("model"))
    ]


def resolve_local_llm(preferred: List[str] | None = None) -> str:
    preferred = preferred or ["llama3.2:3b", "qwen2.5:1.5b", "deepseek-r1:1.5b"]
    available = get_available_ollama_models()
    for m in preferred:
        if m in available:
            return m
    if available:
        return available[0]
    raise RuntimeError(
        "Ollama is not running or no compatible models are installed. "
        "Start Ollama and run: ollama pull llama3.2:3b"
    )


LOCAL_LLM = resolve_local_llm() if get_available_ollama_models() else "llama3.2:3b"


def call_structured_llm(
    response_model: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    model_names: List[str] | None = None,
):
    candidates = model_names or [LOCAL_LLM, "llama3.2:3b", "qwen2.5:1.5b", "deepseek-r1:1.5b"]
    seen, ordered = set(), []
    for m in candidates:
        if m and m not in seen:
            ordered.append(m)
            seen.add(m)

    last_error: Exception | None = None
    for model in ordered:
        try:
            return client.chat.completions.create(
                model=model,
                response_model=response_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt
                        + " Return only valid JSON matching the schema. "
                        + "Every list item must be a plain string or a plain object "
                        + "as specified. Do not add extra keys.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
        except Exception as exc:
            last_error = exc
            print(f"[LLM fallback] {model} failed: {type(exc).__name__}: {exc}")
    raise RuntimeError("All configured Ollama models failed for structured extraction.") from last_error


# ==========================================
# 2. PYDANTIC SCHEMAS (FLAT — small-LLM friendly)
# ==========================================
class DeconstructedIdea(BaseModel):
    """Flat list of feature strings. No nesting — reliable on 3B models."""

    elements: List[str] = Field(
        ...,
        description=(
            "List of 3-6 short technical feature strings extracted from the user's idea. "
            "Example: ['magnetic alignment', 'automatic latch', 'bicycle mounting base']. "
            "Each item MUST be a plain string, NOT an object."
        ),
    )


class DeconstructedPatent(BaseModel):
    patent_id: str = Field(..., description="The patent ID being analyzed.")
    elements: List[str] = Field(
        ...,
        description=(
            "List of 3-6 short technical feature strings extracted from the patent claim. "
            "Each item MUST be a plain string, NOT an object."
        ),
    )


class DiffMatrixResult(BaseModel):
    anticipation_found: bool = Field(
        ...,
        description="True ONLY if a single patent contains every element of the user's idea.",
    )
    anticipating_patent_id: str = Field(
        default="None",
        description="ID of the anticipating patent, or the literal string 'None'.",
    )
    missing_elements_per_patent: Dict[str, List[str]] = Field(
        ...,
        description=(
            "Map of patent_id -> list of user-element strings missing from that patent. "
            "Example: {'US1234567A': ['magnetic alignment'], 'KR102370398B1': []}"
        ),
    )


class ObviousnessResult(BaseModel):
    is_obvious_combination: bool = Field(
        ...,
        description="True if combining two or more references covers all user elements.",
    )
    combined_patent_ids: List[str] = Field(
        default_factory=list,
        description="List of patent ID strings combined under 35 U.S.C. 103.",
    )
    phosita_rationale: str = Field(
        ...,
        description="Why a person having ordinary skill in the art would combine them.",
    )


class ReengineeringReport(BaseModel):
    novelty_status: Literal[
        "Anticipated (102)",
        "Obvious Combination (103)",
        "Potentially Novel",
    ] = Field(..., description="Legal status based on 102 and 103 analysis.")
    uncovered_gaps: List[str] = Field(
        ...,
        description=(
            "Features present in the USER's idea that are NOT found in ANY prior art patent. "
            "Plain strings only."
        ),
    )
    suggested_modifications: List[str] = Field(
        ...,
        description="New technical additions or modifications to design around the cited patents.",
    )
    legal_disclaimer: str = Field(
        default=(
            "AUTOMATED TECHNICAL LANDSCAPE SUMMARY. THIS OUTPUT IS FOR INFORMATIONAL "
            "PURPOSES ONLY AND DOES NOT CONSTITUTE FORMAL LEGAL COUNSEL OR A GUARANTEE OF PATENTABILITY."
        )
    )


# ==========================================
# 3. STATE
# ==========================================
class PatentAgentState(TypedDict):
    user_idea: str
    reranked_patents: List[Dict[str, Any]]
    deconstructed_idea: List[str]
    deconstructed_patents: List[Dict[str, Any]]
    diff_matrix: Dict[str, Any]
    obviousness_analysis: Dict[str, Any]
    final_report: Dict[str, Any]


# ==========================================
# 4. NODES
# ==========================================
def node_element_deconstruction(state: PatentAgentState) -> Dict[str, Any]:
    print("\n--- [Node 1: Element Deconstruction] ---")

    res_idea: DeconstructedIdea = call_structured_llm(
        DeconstructedIdea,
        "Extract 3-6 short technical feature phrases from the user's idea. "
        "Return a JSON object of the form {\"elements\": [\"feature 1\", \"feature 2\", ...]}. "
        "Each element is a plain string.",
        state["user_idea"],
    )
    idea_strings = [s.strip() for s in res_idea.elements if isinstance(s, str) and s.strip()]
    print(f"  User idea elements: {idea_strings}")

    deconstructed_pats = []
    for pat in state["reranked_patents"]:
        res: DeconstructedPatent = call_structured_llm(
            DeconstructedPatent,
            "Extract 3-6 short technical feature phrases from this patent claim. "
            "Return {\"patent_id\": \"...\", \"elements\": [\"...\", \"...\"]}. "
            "Each element is a plain string.",
            f"Patent ID: {pat['patent_id']}\n"
            f"Title: {pat.get('title', '')}\n"
            f"Claim Clause: {pat.get('matched_limitation', '')}",
        )
        # Guarantee the ID matches what the pipeline knows about.
        res.patent_id = pat["patent_id"]
        deconstructed_pats.append(res.model_dump())
        print(f"  Patent {pat['patent_id']} elements: {res.elements}")

    return {
        "deconstructed_idea": idea_strings,
        "deconstructed_patents": deconstructed_pats,
    }


def node_diff_matrix(state: PatentAgentState) -> Dict[str, Any]:
    print("\n--- [Node 2: 1-to-1 Diff Matrix (35 U.S.C. 102)] ---")

    prompt = f"""
    User Idea Elements: {json.dumps(state['deconstructed_idea'])}

    Prior Art Patents: {json.dumps(state['deconstructed_patents'])}

    Task:
    1. For EACH patent, list which user-idea elements are MISSING from it.
    2. Set anticipation_found = true ONLY if one single patent contains ALL user elements.
    3. If anticipation_found is true, set anticipating_patent_id to that patent's ID.
       Otherwise set it to the string "None".

    Return JSON with keys: anticipation_found (bool), anticipating_patent_id (str),
    missing_elements_per_patent (object mapping patent_id -> array of missing element strings).
    """
    res: DiffMatrixResult = call_structured_llm(
        DiffMatrixResult,
        "Compare user idea features against each patent reference strictly.",
        prompt,
    )
    return {"diff_matrix": res.model_dump()}


def node_obviousness_engine(state: PatentAgentState) -> Dict[str, Any]:
    print("\n--- [Node 3: 35 U.S.C. 103 Obviousness Engine] ---")

    if state["diff_matrix"]["anticipation_found"]:
        return {
            "obviousness_analysis": {
                "is_obvious_combination": False,
                "combined_patent_ids": [],
                "phosita_rationale": (
                    "Skipped: invention fully anticipated under 35 U.S.C. 102."
                ),
            }
        }

    prompt = f"""
    User Idea Elements: {json.dumps(state['deconstructed_idea'])}

    Patents Available: {json.dumps(state['deconstructed_patents'])}

    Task: Determine if combining two or more of these patents covers ALL elements
    of the User's Idea, from the perspective of a person having ordinary skill in
    the art (PHOSITA). Return JSON with keys:
      is_obvious_combination (bool),
      combined_patent_ids (array of strings),
      phosita_rationale (string).
    """
    res: ObviousnessResult = call_structured_llm(
        ObviousnessResult,
        "Analyze patent combinations under 35 U.S.C. 103 obviousness standards.",
        prompt,
    )
    return {"obviousness_analysis": res.model_dump()}


def node_safe_reengineering(state: PatentAgentState) -> Dict[str, Any]:
    print("\n--- [Node 4: Safe Re-Engineering & Landscape Framing] ---")

    prompt = f"""
    Analyze the findings to create a final report.

    User Idea Elements: {json.dumps(state['deconstructed_idea'])}
    102 Diff Analysis: {json.dumps(state['diff_matrix'])}
    103 Obviousness Analysis: {json.dumps(state['obviousness_analysis'])}

    Rules:
    - If 102 anticipation_found is true, novelty_status = "Anticipated (102)".
    - Else if 103 is_obvious_combination is true, novelty_status = "Obvious Combination (103)".
    - Otherwise novelty_status = "Potentially Novel".
    - 'uncovered_gaps' must list elements present in the User Idea that NO prior art
      patent contains. Plain strings only.
    - 'suggested_modifications' should propose unique new technical additions to
      escape patent infringement. Plain strings only.

    Return JSON with keys: novelty_status, uncovered_gaps, suggested_modifications.
    """
    res: ReengineeringReport = call_structured_llm(
        ReengineeringReport,
        "Synthesize the final patent landscape and design-around recommendations.",
        prompt,
    )
    return {"final_report": res.model_dump()}


# ==========================================
# 5. GRAPH
# ==========================================
workflow = StateGraph(PatentAgentState)
workflow.add_node("deconstruction", node_element_deconstruction)
workflow.add_node("diff_matrix", node_diff_matrix)
workflow.add_node("obviousness", node_obviousness_engine)
workflow.add_node("reengineering", node_safe_reengineering)
workflow.set_entry_point("deconstruction")
workflow.add_edge("deconstruction", "diff_matrix")
workflow.add_edge("diff_matrix", "obviousness")
workflow.add_edge("obviousness", "reengineering")
workflow.add_edge("reengineering", END)

patent_synthesis_pipeline = workflow.compile()


# ==========================================
# 6. TEST
# ==========================================
if __name__ == "__main__":
    test_state = {
        "user_idea": "A magnetic water bottle holder for bicycles that locks automatically upon alignment",
        "reranked_patents": [
            {
                "patent_id": "KR102370398B1",
                "title": "Closure device for attaching container to bicycle",
                "matched_limitation": "a receiving latch configured to mechanically lock a bottle upon magnetic alignment",
            },
            {
                "patent_id": "US20210169205A1",
                "title": "Device Having Beverage Container Holder",
                "matched_limitation": "a receiving device for detachably fixing the beverage container in place using a mechanical release",
            },
        ],
    }
    output = patent_synthesis_pipeline.invoke(test_state)
    print("\n================ FINAL REPORT ================")
    print(json.dumps(output["final_report"], indent=2))