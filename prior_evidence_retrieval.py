import asyncio
from datetime import datetime
from html import unescape
import re
from typing import Any, Dict, List

import httpx
from pydantic import BaseModel, Field

from input_analysis import PatentQueryAnalysis


# ==========================================
# 1. SCHEMA
# ==========================================
class PriorArtDocument(BaseModel):
    doc_id: str = Field(..., description="Standardized Document Identifier (e.g., US-10123456-B2)")
    title: str
    abstract: str
    cpc_codes: List[str]
    publication_date: str
    is_expired: bool = Field(..., description="Flags if patent term (>20 years) has elapsed.")
    status_tag: str = Field(..., description="'Expired (Prior Art Only)' or 'Active Patent'")
    source_api: str = Field(..., description="Originating authority")
    claims_text: str = Field(default="", description="Full claims text if fetched, else empty.")


# ==========================================
# 2. QUERY BUILDER
# ==========================================
class GooglePatentsQueryBuilder:
    """Translates Step 1 analysis into Google Patents query strings."""

    @staticmethod
    def build_query_a_cpc_keywords(analysis: PatentQueryAnalysis, cpc_codes: List[str]) -> str:
        """Query A: quoted summary + top mechanisms + CPC restrictions."""
        # Use only the two most load-bearing mechanisms to avoid a bag-of-words search.
        mechanisms = analysis.technical_mechanisms[:2] if len(analysis.technical_mechanisms) >= 2 \
            else analysis.technical_mechanisms
        quoted = f'"{analysis.summary}"' if analysis.summary else ""
        cpc_str = " OR ".join(f"CPC={c}" for c in cpc_codes)
        return f"{quoted} {' '.join(mechanisms)} {cpc_str}".strip()

    @staticmethod
    def build_query_b_claims(analysis: PatentQueryAnalysis) -> str:
        """Query B: primary mechanisms only."""
        return " ".join(analysis.technical_mechanisms[:3])

    @staticmethod
    def build_query_c_broad_boolean(analysis: PatentQueryAnalysis) -> str:
        """Query C: summary + legal synonyms."""
        return f"{analysis.summary} {' '.join(analysis.legal_synonyms)}".strip()


# ==========================================
# 3. RETRIEVER
# ==========================================
class AsyncPatentRetriever:
    GOOGLE_PATENTS_URL = "https://patents.google.com/xhr/query"
    GOOGLE_PATENTS_HTML = "https://patents.google.com/patent/{doc_id}/en"

    _CLAIMS_RE = re.compile(
        r'<section[^>]*itemprop="claims"[^>]*>(.*?)</section>',
        re.DOTALL | re.IGNORECASE,
    )
    _TAG_RE = re.compile(r"<[^>]+>")

    @staticmethod
    def _check_expiration(pub_date_str: str) -> tuple[bool, str]:
        try:
            clean = re.sub(r"[^0-9]", "", pub_date_str or "")[:8]
            if len(clean) < 4:
                return False, "Status Unknown"
            year = int(clean[:4])
            if datetime.now().year - year > 20:
                return True, "Expired (Prior Art Only)"
            return False, "Active Patent"
        except (ValueError, TypeError):
            return False, "Status Unknown"

    @staticmethod
    def _clean_text(value: str) -> str:
        return unescape(re.sub(r"<[^>]+>", "", value or "")).strip()

    async def fetch_google_patents_query(
        self,
        client: httpx.AsyncClient,
        search_text: str,
        query_type: str,
    ) -> List[Dict[str, Any]]:
        print(f"  [Async Dispatch] -> Google Patents ({query_type})...")

        params = {"url": f"q=({search_text})&num=5", "exp": ""}
        headers = {"User-Agent": "LocalPatentAgent/1.0", "Accept": "application/json"}

        data: Dict[str, Any] = {}
        for attempt in range(3):
            try:
                response = await client.get(self.GOOGLE_PATENTS_URL, params=params, headers=headers)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.HTTPError, ValueError) as exc:
                if attempt == 2:
                    print(f"  [Google Patents Note] ({query_type}): {exc}")
                    return []

        results: List[Dict[str, Any]] = []
        for cluster in (data.get("results") or {}).get("cluster", []) or []:
            for result in cluster.get("result", []) or []:
                patent = result.get("patent")
                if isinstance(patent, dict):
                    results.append(patent)

        print(f"  [Google Patents] ({query_type}): {len(results)} records.")
        return results[:5]

    async def fetch_claims_text(self, client: httpx.AsyncClient, doc_id: str) -> str:
        """Scrape the claims section from a Google Patents HTML page."""
        if not doc_id or doc_id == "UNKNOWN":
            return ""
        url = self.GOOGLE_PATENTS_HTML.format(doc_id=doc_id)
        try:
            r = await client.get(url, timeout=15.0, follow_redirects=True)
            r.raise_for_status()
        except Exception:
            return ""
        m = self._CLAIMS_RE.search(r.text)
        if not m:
            return ""
        text = unescape(self._TAG_RE.sub(" ", m.group(1)))
        text = re.sub(r"\s+", " ", text).strip()
        return text

    async def execute_parallel_retrieval(
        self,
        analysis: PatentQueryAnalysis,
        cpc_codes: List[str],
    ) -> List[PriorArtDocument]:
        qa = GooglePatentsQueryBuilder.build_query_a_cpc_keywords(analysis, cpc_codes)
        qb = GooglePatentsQueryBuilder.build_query_b_claims(analysis)
        qc = GooglePatentsQueryBuilder.build_query_c_broad_boolean(analysis)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            print("\n--- Initiating Async Google Patents Multi-Query Search ---")
            print(f"  Query A: {qa}")
            print(f"  Query B: {qb}")
            print(f"  Query C: {qc}")
            # Serialized to reduce throttling on the public endpoint.
            nested = [
                await self.fetch_google_patents_query(client, qa, "Query A (CPC + Keywords)"),
                await self.fetch_google_patents_query(client, qb, "Query B (Mechanisms)"),
                await self.fetch_google_patents_query(client, qc, "Query C (Broad Synonyms)"),
            ]

        dedup: Dict[str, PriorArtDocument] = {}
        for raw in (d for sub in nested for d in sub):
            doc_id = raw.get("publication_number") or raw.get("id") or "UNKNOWN"
            if doc_id in dedup:
                continue
            pub_date = str(raw.get("publication_date") or raw.get("date") or "2000-01-01")
            is_expired, status = self._check_expiration(pub_date)
            dedup[doc_id] = PriorArtDocument(
                doc_id=doc_id,
                title=self._clean_text(raw.get("title") or "Untitled"),
                abstract=self._clean_text(
                    raw.get("abstract") or raw.get("snippet") or "No abstract available"
                ),
                cpc_codes=cpc_codes,
                publication_date=pub_date,
                is_expired=is_expired,
                status_tag=status,
                source_api="Google Patents public search",
            )

        docs = list(dedup.values())

        # Enrich with real claim text (concurrently, bounded by the same client).
        if docs:
            print(f"\n  [Enrichment] Fetching full claims text for {len(docs)} docs...")
            async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
                tasks = [self.fetch_claims_text(client, d.doc_id) for d in docs]
                claim_texts = await asyncio.gather(*tasks, return_exceptions=True)
            hits = 0
            for d, ct in zip(docs, claim_texts):
                if isinstance(ct, str) and ct:
                    d.claims_text = ct
                    hits += 1
            print(f"  [Enrichment] Got claims for {hits}/{len(docs)} docs.")

        return docs


# ==========================================
# 4. EXECUTION
# ==========================================
async def main():
    sample = PatentQueryAnalysis(
        summary="A magnetic water bottle holder for bicycles that locks automatically.",
        technical_mechanisms=["magnetic", "latching"],
        legal_synonyms=["receptacle", "vehicle"],
        search_query="magnetic latching receptacle vehicle",
    )
    retriever = AsyncPatentRetriever()
    docs = await retriever.execute_parallel_retrieval(sample, ["B62J", "H01F"])

    print(f"\n--- Retrieved & Deduplicated Prior Art ({len(docs)} Documents) ---")
    for d in docs:
        print(f"ID: {d.doc_id} | Source: {d.source_api}")
        print(f"Title: {d.title}")
        print(f"Pub Date: {d.publication_date} | Status: {d.status_tag}")
        print(f"Abstract: {d.abstract[:200]}...")
        print(f"Claims len: {len(d.claims_text)} chars")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())