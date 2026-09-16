import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from claim_parser import extract_independent_claims, parse_claim_limitations


class PatentReranker:
    def __init__(self, model_name: str = 'BAAI/bge-reranker-large', device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def score_pairs(self, query: str, texts: list[str]) -> list[float]:
        if not texts:
            return []
        pairs = [[query, t] for t in texts]
        inputs = self.tokenizer(
            pairs,
            padding=True,
            truncation=True,          # was: trucation  <-- FIXED
            max_length=512,
            return_tensors="pt",
        ).to(self.device)
        logits = self.model(**inputs, return_dict=True).logits.view(-1).float()
        return torch.sigmoid(logits).cpu().tolist()

    def max_clause_rerank(
        self,
        user_idea: str,
        patent_candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        ranked = []
        for patent in patent_candidates:
            ind_claims = extract_independent_claims(patent.get("text", ""))
            all_limitations: list[str] = []
            for claim in ind_claims:
                parsed = parse_claim_limitations(claim["raw_text"])
                all_limitations.extend(parsed["limitations"])   # was: extent  <-- FIXED

            if not all_limitations:
                fallback = patent.get("text", "")
                if not fallback:
                    continue
                all_limitations = [fallback[:512]]

            clause_scores = self.score_pairs(user_idea, all_limitations)
            if not clause_scores:
                continue

            best_idx = max(range(len(clause_scores)), key=clause_scores.__getitem__)
            max_score = clause_scores[best_idx]

            ranked.append({
                "patent_id": patent.get("id") or patent.get("doc_id"),
                "title": patent.get("title"),
                "max_similarity_score": round(max_score, 4),
                "matched_limitation": all_limitations[best_idx],
                "all_parsed_limitations": all_limitations,
            })

        ranked.sort(key=lambda x: x["max_similarity_score"], reverse=True)
        return ranked[:top_k]