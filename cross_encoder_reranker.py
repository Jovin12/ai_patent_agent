import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from claim_parser import *

class PatentReranker: 
    def __init__(self, model_name: str= 'BAAI/bge-reranker-large', device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        # computer relevance scores for a query paired with multiple target text
        pairs = [[query, text] for text in texts]

        with torch.no_grad():
            inputs = self.tokenizer(
                pairs, 
                padding = True, 
                trucation = True, 
                max_length = 512, 
                return_tensors = "pt",
            ).to(self.device)

            scores = self.model(**inputs, return_dict = True).logits.view(-1).float()
            probabilities = torch.sigmoid(scores).cpu().tolist()

        return probabilities

    def max_clause_rerank(self, user_idea: str, patent_candidates: list[dict], top_k: int = 5) -> list[dict]:
        # score each patent based on its single highest- scoring limitation clause

        ranked_patents = []

        for patent in patent_candidates: 
            ind_claims = extract_independent_claims(patent.get("text", ""))
            all_limitations = []

            for claim in ind_claims: 
                parsed = parse_claim_limitations(claim["raw_text"])
                all_limitations.extent(parsed["limitations"])

            if not all_limitations: 
                all_limiations = [patent.get("text","")[:512]]

            clause_scores = self.score_pairs(user_idea, all_limitations)

            max_score = max(clause_scores) if clause_scores else 0.0
            best_clause_idx = clause_scores.index(max_score) if clause_scores else 0

            ranked_patents.append({
                "patent_id": patent.get("id"),
                "title": patent.get("title"),
                "max_similarity_score": round(max_score, 4),
                "matched_limitation": all_limitations[best_clause_idx],
                "all_parsed_limitations": all_limitations
            })

        ranked_patents.sort(key = lambda x: x['max_similarity_score'], reverse =  True)
        return ranked_patents[:top_k]