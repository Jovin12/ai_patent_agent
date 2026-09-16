import asyncio
from datetime import datetime
from html import unescape
import re
from typing import Any, Dict, List
import httpx
from pydantic import BaseModel, Field

# Import verified schema from Step 1
from input_analysis import PatentQueryAnalysis


# ==========================================
# 1. SCHEMAS FOR RETRIEVED PATENTS
# ==========================================
class PriorArtDocument(BaseModel):
    doc_id: str = Field(
        ...,
        description="Standardized Document Identifier (e.g., US-10123456-B2)",
    )
    title: str
    abstract: str
    cpc_codes: List[str]
    publication_date: str
    is_expired: bool = Field(
        ...,
        description="Flags if patent term (>20 years from filing) has elapsed.",
    )
    status_tag: str = Field(
        ...,
        description="'Expired (Prior Art Only)' or 'Active Patent'",
    )
    source_api: str = Field(
        ..., description="Originating authority (Google Patents public search)"
    )


# ==========================================
# 2. GOOGLE PATENTS QUERY BUILDER
# ==========================================
class GooglePatentsQueryBuilder:
    """Translates Step 1 analysis into Google Patents query strings."""

    @staticmethod
    def build_query_a_cpc_keywords(
        analysis: PatentQueryAnalysis, cpc_codes: List[str]
    ) -> str:
        """Query A: Core Mechanical Keywords + CPC"""
        mechanisms = " ".join(analysis.technical_mechanisms)
        cpc_str = " ".join(cpc_codes)
        return f"{mechanisms} {cpc_str}"

    @staticmethod
    def build_query_b_claims(analysis: PatentQueryAnalysis) -> str:
        """Query B: Technical Mechanisms"""
        return " ".join(analysis.technical_mechanisms)

    @staticmethod
    def build_query_c_broad_boolean(analysis: PatentQueryAnalysis) -> str:
        """Query C: Broad Synonyms & Summary"""
        return f"{analysis.summary} {' '.join(analysis.legal_synonyms)}"


# ==========================================
# 3. ASYNC RETRIEVAL ENGINE (GOOGLE PATENTS PUBLIC SEARCH)
# ==========================================
class AsyncPatentRetriever:

    # Public Google Patents search endpoint; no API key is required.
    GOOGLE_PATENTS_URL = "https://patents.google.com/xhr/query"

    @staticmethod
    def _check_expiration(pub_date_str: str) -> tuple[bool, str]:
        """Determines if a patent is expired based on standard 20-year term limit."""
        try:
            clean_date = pub_date_str.replace("-", "")[:8]
            pub_year = datetime.strptime(clean_date, "%Y%m%d").year
            current_year = datetime.now().year
            if (current_year - pub_year) > 20:
                return True, "Expired (Prior Art Only)"
            return False, "Active Patent"
        except (ValueError, TypeError):
            return False, "Status Unknown"

    async def fetch_google_patents_query(
        self,
        client: httpx.AsyncClient,
        search_text: str,
        query_type: str,
    ) -> List[Dict[str, Any]]:
        """Execute a public Google Patents search and normalize its records."""
        print(f"  [Async Dispatch] -> Google Patents ({query_type})...")

        params = {
            "url": f"q=({search_text})&num=5",
            "exp": "",
        }

        headers = {
            "User-Agent": "LocalPatentAgent/1.0",
            "Accept": "application/json",
        }

        for attempt in range(3):
            try:
                response = await client.get(self.GOOGLE_PATENTS_URL, params=params, headers=headers)
                if response.status_code in {429, 500, 502, 503, 504} and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                response.raise_for_status()
                data = response.json()
                break

            except (httpx.HTTPError, ValueError, KeyError) as error:
                if attempt == 2:
                    print(f"  [Google Patents Connection Note] ({query_type}): {error}")
                    return []

        results: List[Dict[str, Any]] = []
        for cluster in data.get("results", {}).get("cluster", []):
            for result in cluster.get("result", []):
                patent = result.get("patent")
                if patent:
                    results.append(patent)

        print(f"  [Google Patents Success] ({query_type}): Returned {len(results)} records.")
        return results[:5]

    @staticmethod
    def _clean_text(value: str) -> str:
        """Remove highlighting markup included by Google Patents search."""
        return unescape(re.sub(r"<[^>]+>", "", value)).strip()

    async def execute_parallel_retrieval(
        self, analysis: PatentQueryAnalysis, cpc_codes: List[str]
    ) -> List[PriorArtDocument]:
        """Dispatches parallel search requests using async HTTP GET."""
        query_a = GooglePatentsQueryBuilder.build_query_a_cpc_keywords(analysis, cpc_codes)
        query_b = GooglePatentsQueryBuilder.build_query_b_claims(analysis)
        query_c = GooglePatentsQueryBuilder.build_query_c_broad_boolean(analysis)

        async with httpx.AsyncClient(timeout=30.0, follow_redirects=True) as client:
            print("\n--- Initiating Async Google Patents Multi-Query Search ---")

            # Serialize public searches to reduce transient throttling from Google.
            nested_results = [
                await self.fetch_google_patents_query(client, query_a, "Query A (CPC + Keywords)"),
                await self.fetch_google_patents_query(client, query_b, "Query B (Title Keywords)"),
                await self.fetch_google_patents_query(client, query_c, "Query C (Broad Synonyms)"),
            ]

        raw_candidates = [doc for sublist in nested_results for doc in sublist]

        deduplicated_map: Dict[str, PriorArtDocument] = {}

        for raw in raw_candidates:
            # Extract standard ID
            clean_doc_id = raw.get("publication_number") or "UNKNOWN"

            if clean_doc_id not in deduplicated_map:
                pub_date = raw.get("publication_date") or raw.get("date") or "2000-01-01"
                is_expired, status_tag = self._check_expiration(str(pub_date))

                abstract = raw.get("abstract") or raw.get("snippet") or "No abstract available"

                deduplicated_map[clean_doc_id] = PriorArtDocument(
                    doc_id=clean_doc_id,
                    title=self._clean_text(raw.get("title") or "Untitled"),
                    abstract=self._clean_text(abstract),
                    cpc_codes=cpc_codes,
                    publication_date=str(pub_date),
                    is_expired=is_expired,
                    status_tag=status_tag,
                    source_api="Google Patents public search",
                )

        return list(deduplicated_map.values())


# ==========================================
# 4. EXECUTION
# ==========================================
async def main():
    sample_analysis = PatentQueryAnalysis(
        summary="A magnetic water bottle holder for bicycles that locks automatically.",
        technical_mechanisms=[
            "magnetic",
            "latching",
        ],
        legal_synonyms=["receptacle", "vehicle"],
        search_query="magnetic latching receptacle vehicle",
    )
    verified_cpcs = ["B62J", "H01F"]

    retriever = AsyncPatentRetriever()
    candidates = await retriever.execute_parallel_retrieval(
        sample_analysis, verified_cpcs
    )

    print(
        f"\n--- Retrieved & Deduplicated Prior Art ({len(candidates)} Documents) ---"
    )
    for doc in candidates:
        print(f"ID: {doc.doc_id} | Source: {doc.source_api}")
        print(f"Title: {doc.title}")
        print(f"Pub Date: {doc.publication_date} | Status: {doc.status_tag}")
        print(f"Abstract: {doc.abstract}")
        print("-" * 60)


if __name__ == "__main__":
    asyncio.run(main())