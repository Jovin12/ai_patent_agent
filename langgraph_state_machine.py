import json
from typing import Any, Dict, List, Literal, TypedDict
import instructor
from langgraph.graph import END, StateGraph
from openai import OpenAI
from pydantic import BaseModel, Field
from urllib import request

# ==========================================
# 1. SETUP LOCAL OLLAMA INSTRUCTOR CLIENT
# ==========================================
ollama_client = OpenAI(base_url="http://localhost:11434/v1", api_key="ollama")
client = instructor.from_openai(ollama_client, mode=instructor.Mode.JSON)


def get_available_ollama_models() -> List[str]:
    """Return installed Ollama model names, if the local service is reachable."""
    try:
        with request.urlopen("http://localhost:11434/api/tags", timeout=10) as response:
            data = json.load(response)
        models = data.get("models", [])
        return [
            model.get("name") or model.get("model")
            for model in models
            if isinstance(model, dict)
            and (model.get("name") or model.get("model"))
        ]
    except Exception:
        return []


def resolve_local_llm(preferred_models: List[str] | None = None) -> str:
    """Choose the most reliable installed model for structured JSON extraction."""
    preferred = preferred_models or ["llama3.2:3b", "qwen2.5:1.5b", "deepseek-r1:1.5b"]
    available = get_available_ollama_models()

    for model in preferred:
        if model in available:
            return model

    if available:
        return available[0]

    raise RuntimeError(
        "Ollama is not running or no compatible models are installed. "
        "Start Ollama and run: ollama pull llama3.2:3b"
    )


LOCAL_LLM = resolve_local_llm() if get_available_ollama_models() else "llama3.2:3b"


def call_structured_llm(response_model: Any, system_prompt: str, user_prompt: str, *, model_names: List[str] | None = None):
    """Retry the structured call across installed Ollama models to avoid schema drift."""
    candidates = model_names or [LOCAL_LLM, "llama3.2:3b", "qwen2.5:1.5b", "deepseek-r1:1.5b"]
    deduped = []
    seen = set()
    for model in candidates:
        if model and model not in seen:
            deduped.append(model)
            seen.add(model)

    last_error = None
    for model in deduped:
        try:
            return client.chat.completions.create(
                model=model,
                response_model=response_model,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt + " Return only valid JSON matching the schema with no prose or markdown.",
                    },
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.0,
            )
        except Exception as exc:
            last_error = exc
            print(f"[LLM fallback] model {model} failed: {type(exc).__name__}: {exc}")

    raise RuntimeError("All configured Ollama models failed for structured extraction.") from last_error


# ==========================================
# 2. PYDANTIC SCHEMAS (STRUCTURED FOR SMALL LLMS)
# ==========================================
class FeatureElement(BaseModel):
    feature: str = Field(
        ...,
        description="A single technical feature or component name (e.g., 'magnetic alignment').",
    )


class PatentElement(BaseModel):
    title: str = Field(
        ...,
        description="Short name of the mechanical component (e.g., 'Receiving Latch').",
    )
    description: str = Field(
        ..., description="Brief functional explanation of the component."
    )


class DeconstructedIdea(BaseModel):
    elements: List[FeatureElement] = Field(
        ...,
        description="List of feature objects extracted from the user's idea.",
    )


class DeconstructedPatent(BaseModel):
    patent_id: str
    elements: List[PatentElement] = Field(
        ...,
        description="List of detailed structural elements extracted from patent claims.",
    )


class DiffMatrixResult(BaseModel):
    anticipation_found: bool = Field(
        ...,
        description="True ONLY if a single patent contains every single element of the user's idea.",
    )
    anticipating_patent_id: str = Field(
        default="None",
        description="ID of the anticipating patent, if anticipation_found is True.",
    )
    missing_elements_per_patent: Dict[str, List[str]] = Field(
        ..., description="User elements missing in each individual patent reference."
    )


class ObviousnessResult(BaseModel):
    is_obvious_combination: bool = Field(
        ...,
        description="True if combining Reference A + Reference B covers all user elements.",
    )
    combined_patent_ids: List[str] = Field(
        default_factory=list, description="Patents combined under 35 U.S.C. 103."
    )
    phosita_rationale: str = Field(
        ...,
        description="Why a person having ordinary skill in the art (PHOSITA) would combine them.",
    )


class ReengineeringReport(BaseModel):
    novelty_status: Literal["Anticipated (102)", "Obvious Combination (103)", "Potentially Novel"] = Field(
        ...,
        description="Legal status based on 102 and 103 analysis.",
    )
    uncovered_gaps: List[str] = Field(
        ...,
        description="Features present in the USER's idea that are NOT found in ANY prior art patents.",
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
# 3. LANGGRAPH STATE DEFINITION
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
# 4. LANGGRAPH NODES
# ==========================================
def node_element_deconstruction(state: PatentAgentState) -> Dict[str, Any]:
    """Node 1: Deconstructs user idea and patents into structured components."""
    print("\n--- [Node 1: Element Deconstruction] ---")

    res_idea: DeconstructedIdea = call_structured_llm(
        DeconstructedIdea,
        "Extract key mechanical or technical features from the user's idea as short noun phrases.",
        state["user_idea"],
    )

    extracted_idea_strings = [item.feature for item in res_idea.elements]

    deconstructed_pats = []
    for pat in state["reranked_patents"]:
        res_pat: DeconstructedPatent = call_structured_llm(
            DeconstructedPatent,
            f"Extract technical features for patent ID: {pat['patent_id']}.",
            f"Title: {pat['title']}\nClaim Clause: {pat['matched_limitation']}",
        )
        deconstructed_pats.append(res_pat.model_dump())

    return {
        "deconstructed_idea": extracted_idea_strings,
        "deconstructed_patents": deconstructed_pats,
    }


def node_diff_matrix(state: PatentAgentState) -> Dict[str, Any]:
    """Node 2: Evaluates 35 U.S.C. 102 (Anticipation / Novelty Conflict)."""
    print("\n--- [Node 2: 1-to-1 Diff Matrix (35 U.S.C. 102)] ---")

    prompt = f"""
    User Idea Elements: {state['deconstructed_idea']}
    Prior Art Patents: {json.dumps(state['deconstructed_patents'])}

    Task:
    1. Check if ANY single patent contains 100% of the User Idea Elements.
    2. List missing elements per patent ID.
    """

    res: DiffMatrixResult = call_structured_llm(
        DiffMatrixResult,
        "Compare user idea features against each patent reference strictly.",
        prompt,
    )
    return {"diff_matrix": res.model_dump()}


def node_obviousness_engine(state: PatentAgentState) -> Dict[str, Any]:
    """Node 3: Evaluates 35 U.S.C. 103 (Multi-Patent Obviousness Combinations)."""
    print("\n--- [Node 3: 35 U.S.C. 103 Obviousness Engine] ---")

    if state["diff_matrix"]["anticipation_found"]:
        return {
            "obviousness_analysis": {
                "is_obvious_combination": False,
                "combined_patent_ids": [],
                "phosita_rationale": "Skipped because invention is fully anticipated under 35 U.S.C. 102.",
            }
        }

    prompt = f"""
    User Elements: {state['deconstructed_idea']}
    Patents Available: {json.dumps(state['deconstructed_patents'])}

    Task:
    Determine if combining two or more of these patents covers all elements of the User's Idea.
    """

    res: ObviousnessResult = call_structured_llm(
        ObviousnessResult,
        "Analyze patent combinations under 35 U.S.C. 103 obviousness standards.",
        prompt,
    )
    return {"obviousness_analysis": res.model_dump()}


def node_safe_reengineering(state: PatentAgentState) -> Dict[str, Any]:
    """Node 4: Landscape Framing & Workaround Suggestions."""
    print("\n--- [Node 4: Safe Re-Engineering & Landscape Framing] ---")

    prompt = f"""
    Analyze the findings to create a final report:
    User Idea Elements: {state['deconstructed_idea']}
    102 Diff Analysis: {json.dumps(state['diff_matrix'])}
    103 Obviousness Analysis: {json.dumps(state['obviousness_analysis'])}

    Rules:
    - If 102 anticipation is True, status is 'Anticipated (102)'.
    - If 103 combination is True, status is 'Obvious Combination (103)'.
    - Otherwise, status is 'Potentially Novel'.
    - 'uncovered_gaps' must list elements present in the User Idea that NO prior art patent contains.
    - 'suggested_modifications' should propose unique new technical additions to escape patent infringement.
    """

    res: ReengineeringReport = call_structured_llm(
        ReengineeringReport,
        "Synthesize the final patent landscape and design-around recommendations.",
        prompt,
    )
    return {"final_report": res.model_dump()}


# ==========================================
# 5. BUILD & COMPILE LANGGRAPH FLOW
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
# 6. EXECUTION PIPELINE TEST
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