import re

# Regex for numbered claim starts at line beginning (e.g., "1. ", "12. ")
_CLAIM_START = re.compile(r'(?m)^\s*(\d+)\.\s+')
# Dependent-claim references: "of claim 1", "according to claim 2", "claims 1-3", "any of claims 1 to 4"
_DEP_RE = re.compile(
    r'\b(?:of|according to|as in|as claimed in|in)\s+claim[s]?\s+\d+',
    re.IGNORECASE,
)


def extract_independent_claims(patent_text: str) -> list[dict]:
    """Extract independent claims from full patent text."""
    if not patent_text:
        return []

    # Find every claim-start position, then slice between them.
    starts = [(m.start(), m.group(1)) for m in _CLAIM_START.finditer(patent_text)]
    if not starts:
        return []

    independent = []
    for i, (pos, num) in enumerate(starts):
        end = starts[i + 1][0] if i + 1 < len(starts) else len(patent_text)
        body = patent_text[pos:end]
        # Strip the leading "N. "
        body = _CLAIM_START.sub('', body, count=1).strip()

        if _DEP_RE.search(body):
            continue  # dependent claim — skip
        independent.append({"claim_id": num, "raw_text": body})

    return independent


# Transition phrases, ordered longest-first so "consisting essentially of" wins over "consisting of"
_TRANSITIONS = [
    "consisting essentially of",
    "consisting of",
    "characterized in that",
    "comprising",
    "including",
    "having",
    "wherein",
]
_TRANSITION_RE = re.compile(
    r'\b(' + '|'.join(re.escape(t) for t in _TRANSITIONS) + r')\b',
    re.IGNORECASE,
)
# Split limitations on semicolons OR on ", wherein" / ", whereby" boundaries
_LIMIT_SPLIT = re.compile(r';|,\s*(?=wherein\b|whereby\b)', re.IGNORECASE)


def parse_claim_limitations(claim_text: str) -> dict:
    """Break an independent claim into preamble, transition, and limitations."""
    if not claim_text:
        return {"preamble": "", "transition": "", "limitations": []}

    match = _TRANSITION_RE.search(claim_text)
    if match:
        preamble = claim_text[: match.start()].strip()
        transition = match.group(0).strip().lower()
        body = claim_text[match.end():].strip()
    else:
        # Fallback: treat everything up to the first comma as preamble
        comma = claim_text.find(',')
        cut = comma if comma != -1 else min(100, len(claim_text))
        preamble = claim_text[:cut].strip()
        transition = ""
        body = claim_text[cut:].strip()

    limitations = [p.strip() for p in _LIMIT_SPLIT.split(body) if p.strip()]
    return {"preamble": preamble, "transition": transition, "limitations": limitations}