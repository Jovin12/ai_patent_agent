import re 
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_independent_claims(patent_text: str) -> list[dict]:
    # extract independent claims form full patent text
    claim_blocks = re.split(r'\n(?=\d+\.\s+)', patent_text.strip())
    independent_claims = []

    for block in claim_blocks: 
        if not block.strip(): 
            continue

        # check if it references previous claims
        is_dependent = re.search(r'\b(of|according to)\s+claim\s+\d+\b', block, re.IGNORECASE)
        if not is_dependent:
            match = re.match(r'^(\d+)\.\s+(.*)' , block, re.DOTALL)
            if match: 
                claim_num, claim_body = match.groups()
                independent_claims.append({
                    "claim_id": claim_num,
                    "raw_text": claim_body.strip()
                })
    return independent_claims

def parse_claim_limitations(claim_text: str) -> dict: 
    # breaks ind claim into preamble, transitiona nd discrete limit

    transitions = r'\b(comprising|consisting of|consisting essentially of|characterized in that|having|including)\b'
    match = re.search(transitions, claim_text, re.IGNORECASE)
    if match: 
        preamble = claim_text[:match.start()].strip()
        transition = match.group(0).strip()
        body = claim_text[match.end():].strip()
    else: 
        preamble = claim_text[:100]
        transition = ""
        body = claim_text[100:]


    # parse body into limitations using semi-colons 
    raw_limitations = re.split(r';|\.\s*$', body)

    limitations = []
    for lim in raw_limitations: 
        cleaned = lim.strip()
        if cleaned: 
            limitations.append(cleaned)

    return{
        "preamble": preamble, 
        "transition": transition, 
        "limitations": limitations
    }