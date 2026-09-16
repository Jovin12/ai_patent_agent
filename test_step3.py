import re
import spacy
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load lightweight spaCy model
nlp = spacy.load("en_core_web_sm")


# ==========================================
# 1. PARSER FUNCTIONS
# ==========================================
def extract_independent_claims(patent_text: str) -> list[dict]:
    """Separates independent claims from dependent claims."""
    claim_blocks = re.split(r'\n(?=\d+\.\s+)', patent_text.strip())
    independent_claims = []

    for block in claim_blocks:
        if not block.strip():
            continue

        # Ignore dependent claims referencing prior claims
        is_dependent = re.search(
            r'\b(of|according to)\s+claim\s+\d+\b', block, re.IGNORECASE
        )
        if not is_dependent:
            match = re.match(r'^(\d+)\.\s+(.*)', block, re.DOTALL)
            if match:
                claim_num, claim_body = match.groups()
                independent_claims.append(
                    {"claim_id": claim_num, "raw_text": claim_body.strip()}
                )
    return independent_claims


def parse_claim_limitations(claim_text: str) -> dict:
    """Extracts Preamble, Transition, and itemized Limitations."""
    transitions = r'\b(comprising|consisting of|consisting essentially of|characterized in that|having|including)\b'
    match = re.search(transitions, claim_text, re.IGNORECASE)

    if match:
        preamble = claim_text[: match.start()].strip()
        transition = match.group(0).strip()
        body = claim_text[match.end() :].strip()
    else:
        preamble = claim_text[:100]
        transition = ""
        body = claim_text[100:]

    # Split body into clauses by semicolons or period endings
    raw_limitations = re.split(r';|\.\s*$', body)
    limitations = [lim.strip() for lim in raw_limitations if lim.strip()]

    return {
        "preamble": preamble,
        "transition": transition,
        "limitations": limitations,
    }


# ==========================================
# 2. LOCAL CROSS-ENCODER MODEL
# ==========================================
class LocalPatentReranker:

    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-large",
        device: str | None = None,
    ):
        # Default to CPU to conserve GPU VRAM for Ollama
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading {model_name} onto {self.device.upper()}...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        )
        self.model.to(self.device)
        self.model.eval()

    def score_clauses(self, query: str, clauses: list[str]) -> list[float]:
        """Calculates relevance score for (user_idea, claim_clause) pairs."""
        pairs = [[query, clause] for clause in clauses]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(self.device)

            scores = (
                self.model(**inputs, return_dict=True).logits.view(-1).float()
            )
            probabilities = torch.sigmoid(scores).cpu().tolist()

        return (
            [probabilities]
            if isinstance(probabilities, float)
            else probabilities
        )


# ==========================================
# 3. DUMMY DATA & TEST EXECUTION
# ==========================================
USER_IDEA = "A magnetic water bottle holder for bicycles that automatically latches into place."

DUMMY_PATENT_TEXT = """
1. A closure apparatus for securing a liquid receptacle to a vehicle frame comprising:
a base station configured for mounting onto a tube of a two-wheeled bicycle;
a first magnetic component embedded within said base station;
a receiving latch configured to mechanically lock a bottle upon magnetic alignment;
and a secondary manual release trigger.

2. The closure apparatus of claim 1, wherein the liquid receptacle is made of extruded aluminum.

3. An adjustable electronic sensor system for a vehicle comprising:
a wireless transceiver module;
an accelerometer mounted to a steering shaft;
and an LED indicator status assembly.
"""

if __name__ == "__main__":
    print(f"\n[User Idea Target]: {USER_IDEA}\n")

    # Step A: Parsing Claims
    ind_claims = extract_independent_claims(DUMMY_PATENT_TEXT)
    print(f"--- Extracted {len(ind_claims)} Independent Claims ---")

    all_parsed_clauses = []
    for claim in ind_claims:
        parsed = parse_claim_limitations(claim["raw_text"])
        print(f"\nClaim {claim['claim_id']} Preamble: '{parsed['preamble']}'")
        print(f"Transition Keyword: '{parsed['transition']}'")
        print("Parsed Limitations:")
        for idx, lim in enumerate(parsed["limitations"]):
            print(f"  [{idx+1}] {lim}")
            all_parsed_clauses.append((claim["claim_id"], lim))

    # Step B: Reranking Limitations
    print("\n--- Running BGE Cross-Encoder Clause Reranking ---")
    reranker = LocalPatentReranker(device="cpu")

    just_clauses = [item[1] for item in all_parsed_clauses]
    scores = reranker.score_clauses(USER_IDEA, just_clauses)

    # Output Scored Limitations
    results = []
    for (claim_id, clause), score in zip(
        all_parsed_clauses, scores, strict=False
    ):
        results.append(
            {"claim_id": claim_id, "clause": clause, "similarity_score": score}
        )

    results.sort(key=lambda x: x["similarity_score"], reverse=True)

    print("\n--- Reranked Clauses (Highest Similarity to Lowest) ---")
    for r in results:
        print(
            f"Score: {r['similarity_score']:.4f} | Claim {r['claim_id']}: {r['clause']}"
        ) # successfully tested test_step3