"""
Test suite for the local AI patent agent.

Run modes:
    pytest -m unit          # fast, offline
    pytest -m model         # loads HF cross-encoder (GPU recommended)
    pytest -m llm           # requires Ollama
    pytest -m network       # requires internet
    pytest                 # everything (slow)
"""
import asyncio
import json
import re
from typing import Any, Dict, List

import pytest


# ==================================================================
# 1. claim_parser.py
# ==================================================================
@pytest.mark.unit
class TestClaimParser:
    def test_import(self):
        import claim_parser  # noqa: F401

    def test_extract_independent_claims_basic(self, sample_patent_text):
        from claim_parser import extract_independent_claims

        claims = extract_independent_claims(sample_patent_text)
        ids = [c["claim_id"] for c in claims]
        # Claims 1, 4, 6 are independent; 2,3,5 are dependent.
        assert ids == ["1", "4", "6"], f"unexpected independent claims: {ids}"

    def test_extract_independent_claims_handles_empty(self):
        from claim_parser import extract_independent_claims

        assert extract_independent_claims("") == []
        assert extract_independent_claims("no numbered claims here") == []

    def test_extract_ignores_dependent_variants(self):
        from claim_parser import extract_independent_claims

        text = (
            "1. A widget comprising a body.\n"
            "2. The widget of claim 1, wherein the body is metal.\n"
            "3. A widget according to claim 1, further comprising a lid.\n"
            "4. A widget as claimed in claim 2, wherein the lid is plastic.\n"
            "5. A widget as in any one of claims 1 to 4.\n"
        )
        claims = extract_independent_claims(text)
        assert [c["claim_id"] for c in claims] == ["1"]

    def test_extract_handles_multiline_claims(self):
        from claim_parser import extract_independent_claims

        text = (
            "1. A device comprising:\n"
            "   a first element;\n"
            "   a second element; and\n"
            "   a third element.\n"
            "2. The device of claim 1, wherein the first element is round.\n"
        )
        claims = extract_independent_claims(text)
        assert len(claims) == 1
        assert "third element" in claims[0]["raw_text"]

    def test_parse_claim_limitations_preamble_and_transition(self, sample_claim_block):
        from claim_parser import parse_claim_limitations

        parsed = parse_claim_limitations(sample_claim_block)
        assert parsed["preamble"].lower().startswith("a magnetic bottle holder")
        assert parsed["transition"].lower() == "comprising"
        assert len(parsed["limitations"]) >= 4
        joined = " ".join(parsed["limitations"]).lower()
        for term in ["mounting base", "receptacle", "magnetic alignment", "latching mechanism"]:
            assert term in joined

    def test_parse_claim_limitations_no_transition(self):
        from claim_parser import parse_claim_limitations

        parsed = parse_claim_limitations("A simple widget with a body and a lid")
        assert parsed["transition"] == ""
        assert parsed["preamble"]
        # Must not raise, must return a list
        assert isinstance(parsed["limitations"], list)

    def test_parse_claim_limitations_empty(self):
        from claim_parser import parse_claim_limitations

        parsed = parse_claim_limitations("")
        assert parsed == {"preamble": "", "transition": "", "limitations": []}

    def test_transition_prefers_longest_match(self):
        from claim_parser import parse_claim_limitations

        text = "A composition consisting essentially of water and salt."
        parsed = parse_claim_limitations(text)
        assert parsed["transition"] == "consisting essentially of"

    def test_splits_on_wherein_comma(self):
        from claim_parser import parse_claim_limitations

        text = (
            "A widget comprising a body, wherein the body is metal, "
            "wherein the metal is steel."
        )
        parsed = parse_claim_limitations(text)
        assert len(parsed["limitations"]) >= 2


# ==================================================================
# 2. cross_encoder_reranker.py  (structural + mocked)
# ==================================================================
@pytest.mark.unit
class TestRerankerStructure:
    """No model download, no GPU. Mocks the transformers bits."""

    def test_max_clause_rerank_uses_extend_not_extent(self, monkeypatch, sample_patent_candidates, sample_user_idea):
        """
        Regression: original code called `all_limitations.extent(...)` (typo)
        which would AttributeError. Verify the fixed version uses `.extend`.
        """
        import cross_encoder_reranker as cer

        instance = cer.PatentReranker.__new__(cer.PatentReranker)  # skip __init__

        # Fake score_pairs so we don't load the real cross-encoder.
        def fake_score_pairs(query, texts):
            assert isinstance(texts, list) and texts
            return [min(1.0, 0.1 * (i + 1)) for i in range(len(texts))]

        instance.score_pairs = fake_score_pairs  # type: ignore[assignment]

        out = instance.max_clause_rerank(sample_user_idea, sample_patent_candidates, top_k=2)
        assert len(out) == 2
        # Sorted descending by score
        assert out[0]["max_similarity_score"] >= out[1]["max_similarity_score"]
        for row in out:
            assert "patent_id" in row
            assert "matched_limitation" in row
            assert isinstance(row["all_parsed_limitations"], list)
            assert row["all_parsed_limitations"], "limitations list must not be empty"

    def test_max_clause_rerank_empty_candidates(self):
        import cross_encoder_reranker as cer

        instance = cer.PatentReranker.__new__(cer.PatentReranker)
        instance.score_pairs = lambda q, t: []  # type: ignore[assignment]
        assert instance.max_clause_rerank("idea", [], top_k=5) == []

    def test_max_clause_rerank_falls_back_to_text_snippet(self):
        """Patent with unparseable claims should use its first 512 chars."""
        import cross_encoder_reranker as cer

        instance = cer.PatentReranker.__new__(cer.PatentReranker)
        captured = {}

        def fake_score_pairs(query, texts):
            captured["texts"] = texts
            return [0.5] * len(texts)

        instance.score_pairs = fake_score_pairs  # type: ignore[assignment]
        candidates = [{"id": "X", "title": "T", "text": "Just prose, no numbered claims."}]
        out = instance.max_clause_rerank("idea", candidates, top_k=1)
        assert out and out[0]["patent_id"] == "X"
        assert captured["texts"], "should have fallen back to raw text"
        assert captured["texts"][0].startswith("Just prose")


# ==================================================================
# 3. input_analysis.py  (schema + mocked LLM)
# ==================================================================
@pytest.mark.unit
class TestInputAnalysis:
    def test_pydantic_schema_valid(self):
        from input_analysis import PatentQueryAnalysis

        obj = PatentQueryAnalysis(
            summary="A magnetic bottle holder.",
            technical_mechanisms=["magnetic", "latching"],
            legal_synonyms=["receptacle", "vehicle"],
            search_query="magnetic latching receptacle vehicle",
        )
        dumped = obj.model_dump()
        assert set(dumped) == {"summary", "technical_mechanisms", "legal_synonyms", "search_query"}

    def test_sample_cpc_data_shape(self):
        from input_analysis import SAMPLE_CPC_DATA

        assert isinstance(SAMPLE_CPC_DATA, list) and SAMPLE_CPC_DATA
        for item in SAMPLE_CPC_DATA:
            assert set(item) == {"id", "description"}
            assert item["id"] and item["description"]

    def test_ensure_seeded_is_idempotent(self):
        from input_analysis import SAMPLE_CPC_DATA, ensure_seeded

        class FakeCollection:
            def __init__(self):
                self._count = 0
                self.add_calls = 0

            def count(self):
                return self._count

            def add(self, **kwargs):
                self.add_calls += 1
                self._count += len(kwargs["ids"])

        col = FakeCollection()
        ensure_seeded(col)
        ensure_seeded(col)
        assert col.add_calls == 1, "should only seed when collection is empty"
        assert col.count() == len(SAMPLE_CPC_DATA)


# ==================================================================
# 4. prior_evidence_retrieval.py  (offline: expiration + query builder)
# ==================================================================
@pytest.mark.unit
class TestPriorEvidenceRetrieval:
    def test_expiration_flags_old_patent(self):
        from prior_evidence_retrieval import AsyncPatentRetriever

        is_expired, tag = AsyncPatentRetriever._check_expiration("1995-06-01")
        assert is_expired is True
        assert "Expired" in tag

    def test_expiration_flags_recent_patent(self):
        from prior_evidence_retrieval import AsyncPatentRetriever

        is_expired, tag = AsyncPatentRetriever._check_expiration("2024-01-15")
        assert is_expired is False
        assert "Active" in tag

    def test_expiration_handles_garbage(self):
        from prior_evidence_retrieval import AsyncPatentRetriever

        for bad in ["", "not-a-date", None, "abcd-ef-gh"]:
            is_expired, tag = AsyncPatentRetriever._check_expiration(bad)
            assert is_expired is False
            assert tag == "Status Unknown"

    def test_clean_text_strips_html_and_unescapes(self):
        from prior_evidence_retrieval import AsyncPatentRetriever

        raw = "<b>Magnetic</b> &amp; <i>latching</i> holder"
        cleaned = AsyncPatentRetriever._clean_text(raw)
        assert cleaned == "Magnetic & latching holder"

    def test_query_builder_includes_cpc_and_mechanisms(self):
        from input_analysis import PatentQueryAnalysis
        from prior_evidence_retrieval import GooglePatentsQueryBuilder

        analysis = PatentQueryAnalysis(
            summary="magnetic bottle holder",
            technical_mechanisms=["magnetic", "latching"],
            legal_synonyms=["receptacle"],
            search_query="magnetic latching receptacle",
        )
        qa = GooglePatentsQueryBuilder.build_query_a_cpc_keywords(analysis, ["B62J", "H01F"])
        assert "magnetic" in qa and "latching" in qa
        assert "B62J" in qa and "H01F" in qa

        qb = GooglePatentsQueryBuilder.build_query_b_claims(analysis)
        assert qb == "magnetic latching"

        qc = GooglePatentsQueryBuilder.build_query_c_broad_boolean(analysis)
        assert "magnetic bottle holder" in qc and "receptacle" in qc

    def test_prior_art_document_schema(self):
        from prior_evidence_retrieval import PriorArtDocument

        doc = PriorArtDocument(
            doc_id="US-1234567-A",
            title="Holder",
            abstract="A holder.",
            cpc_codes=["B62J"],
            publication_date="2020-01-01",
            is_expired=False,
            status_tag="Active Patent",
            source_api="Google Patents public search",
        )
        assert doc.doc_id == "US-1234567-A"


# ==================================================================
# 5. langgraph_state_machine.py  (schemas + graph shape + mocked LLM)
# ==================================================================
@pytest.mark.unit
class TestLangGraphSchemas:
    def test_deconstructed_idea_schema(self):
        from langgraph_state_machine import DeconstructedIdea

        obj = DeconstructedIdea(elements=[{"feature": "magnetic alignment"}])
        assert obj.elements[0].feature == "magnetic alignment"

    def test_reengineering_report_status_literal(self):
        from langgraph_state_machine import ReengineeringReport

        for status in ["Anticipated (102)", "Obvious Combination (103)", "Potentially Novel"]:
            r = ReengineeringReport(
                novelty_status=status,
                uncovered_gaps=[],
                suggested_modifications=[],
            )
            assert r.novelty_status == status
            assert "NOT CONSTITUTE FORMAL LEGAL COUNSEL" in r.legal_disclaimer

    def test_reengineering_report_rejects_bad_status(self):
        from langgraph_state_machine import ReengineeringReport
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            ReengineeringReport(
                novelty_status="Totally Novel Trust Me",
                uncovered_gaps=[],
                suggested_modifications=[],
            )

    def test_diff_matrix_default_patent_id(self):
        from langgraph_state_machine import DiffMatrixResult

        r = DiffMatrixResult(
            anticipation_found=False,
            missing_elements_per_patent={"US1": ["A"]},
        )
        assert r.anticipating_patent_id == "None"

    def test_graph_is_compiled(self):
        import langgraph_state_machine as lgsm

        assert lgsm.patent_synthesis_pipeline is not None
        # LangGraph compiled object exposes .invoke
        assert hasattr(lgsm.patent_synthesis_pipeline, "invoke")

    def test_graph_nodes_present(self):
        import langgraph_state_machine as lgsm

        # If the graph object exposes its nodes, sanity check names.
        graph = getattr(lgsm.patent_synthesis_pipeline, "get_graph", None)
        if callable(graph):
            nodes = set(graph().nodes.keys())
            for name in ["deconstruction", "diff_matrix", "obviousness", "reengineering"]:
                assert any(name in n for n in nodes), f"missing node {name} in {nodes}"


@pytest.mark.unit
class TestLangGraphNodesWithMockedLLM:
    """Exercise node logic without hitting Ollama."""

    def test_node_obviousness_short_circuits_on_anticipation(self, monkeypatch):
        import langgraph_state_machine as lgsm

        called = {"n": 0}

        def fake_call(*args, **kwargs):
            called["n"] += 1
            raise AssertionError("LLM should not be called when anticipated")

        monkeypatch.setattr(lgsm, "call_structured_llm", fake_call)

        state = {
            "user_idea": "x",
            "reranked_patents": [],
            "deconstructed_idea": ["A", "B"],
            "deconstructed_patents": [],
            "diff_matrix": {"anticipation_found": True, "anticipating_patent_id": "US1",
                            "missing_elements_per_patent": {}},
            "obviousness_analysis": {},
            "final_report": {},
        }
        out = lgsm.node_obviousness_engine(state)
        assert out["obviousness_analysis"]["is_obvious_combination"] is False
        assert called["n"] == 0

    def test_node_deconstruction_with_fake_llm(self, monkeypatch, sample_reranked_patents):
        import langgraph_state_machine as lgsm

        def fake_call(response_model, system_prompt, user_prompt, **kwargs):
            name = response_model.__name__
            if name == "DeconstructedIdea":
                return response_model(elements=[{"feature": "magnetic alignment"},
                                                {"feature": "automatic latch"}])
            if name == "DeconstructedPatent":
                return response_model(
                    patent_id="p",
                    elements=[{"title": "latch", "description": "locks the bottle"}],
                )
            raise AssertionError(f"unexpected response_model {name}")

        monkeypatch.setattr(lgsm, "call_structured_llm", fake_call)

        state = {
            "user_idea": "magnetic bottle holder",
            "reranked_patents": sample_reranked_patents,
            "deconstructed_idea": [],
            "deconstructed_patents": [],
            "diff_matrix": {},
            "obviousness_analysis": {},
            "final_report": {},
        }
        out = lgsm.node_element_deconstruction(state)
        assert out["deconstructed_idea"] == ["magnetic alignment", "automatic latch"]
        assert len(out["deconstructed_patents"]) == len(sample_reranked_patents)
        assert out["deconstructed_patents"][0]["patent_id"] == "p"

    def test_node_safe_reengineering_with_fake_llm(self, monkeypatch):
        import langgraph_state_machine as lgsm

        def fake_call(response_model, system_prompt, user_prompt, **kwargs):
            assert response_model.__name__ == "ReengineeringReport"
            return response_model(
                novelty_status="Potentially Novel",
                uncovered_gaps=["self-locking hinge"],
                suggested_modifications=["Add a reed switch"],
            )

        monkeypatch.setattr(lgsm, "call_structured_llm", fake_call)

        state = {
            "user_idea": "x",
            "reranked_patents": [],
            "deconstructed_idea": ["A"],
            "deconstructed_patents": [],
            "diff_matrix": {"anticipation_found": False},
            "obviousness_analysis": {"is_obvious_combination": False},
            "final_report": {},
        }
        out = lgsm.node_safe_reengineering(state)
        assert out["final_report"]["novelty_status"] == "Potentially Novel"
        assert out["final_report"]["suggested_modifications"] == ["Add a reed switch"]


# ==================================================================
# 6. Integration: full pipeline with stubbed LLM + real claim parser
# ==================================================================
@pytest.mark.unit
class TestIntegrationWithStubs:
    def test_end_to_end_pipeline_stubbed(self, monkeypatch, sample_reranked_patents):
        import langgraph_state_machine as lgsm

        responses = {
            "DeconstructedIdea": lambda m: m(elements=[
                {"feature": "magnetic alignment"},
                {"feature": "automatic locking"},
            ]),
            "DeconstructedPatent": lambda m: m(
                patent_id="stub",
                elements=[{"title": "latch", "description": "mechanical lock"}],
            ),
            "DiffMatrixResult": lambda m: m(
                anticipation_found=False,
                anticipating_patent_id="None",
                missing_elements_per_patent={"stub": ["automatic locking"]},
            ),
            "ObviousnessResult": lambda m: m(
                is_obvious_combination=True,
                combined_patent_ids=["KR102370398B1", "US20210169205A1"],
                phosita_rationale="Both teach magnetic alignment and mechanical locking.",
            ),
            "ReengineeringReport": lambda m: m(
                novelty_status="Obvious Combination (103)",
                uncovered_gaps=["reed-switch trigger"],
                suggested_modifications=["Add a Hall-effect sensor for release detection."],
            ),
        }

        def fake_call(response_model, system_prompt, user_prompt, **kwargs):
            return responses[response_model.__name__](response_model)

        monkeypatch.setattr(lgsm, "call_structured_llm", fake_call)

        state = {
            "user_idea": "magnetic bottle holder for bicycles",
            "reranked_patents": sample_reranked_patents,
            "deconstructed_idea": [],
            "deconstructed_patents": [],
            "diff_matrix": {},
            "obviousness_analysis": {},
            "final_report": {},
        }
        out = lgsm.patent_synthesis_pipeline.invoke(state)
        assert out["final_report"]["novelty_status"] == "Obvious Combination (103)"
        assert out["final_report"]["uncovered_gaps"] == ["reed-switch trigger"]
        assert "legal_disclaimer" in out["final_report"]


# ==================================================================
# 7. Model tests (real HF cross-encoder, GPU recommended)
# ==================================================================
@pytest.mark.model
class TestRerankerRealModel:
    @pytest.fixture(scope="class")
    def reranker(self):
        from cross_encoder_reranker import PatentReranker
        # bge-reranker-base is much lighter than large; swap if you have VRAM.
        return PatentReranker(model_name="BAAI/bge-reranker-base")

    def test_score_pairs_returns_probabilities(self, reranker):
        scores = reranker.score_pairs(
            "magnetic bottle holder",
            ["a magnetic bottle holder for bicycles", "a hydraulic brake system"],
        )
        assert len(scores) == 2
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_score_pairs_empty_input(self, reranker):
        assert reranker.score_pairs("q", []) == []

    def test_relevance_ordering(self, reranker):
        scores = reranker.score_pairs(
            "magnetic water bottle holder for bicycles",
            [
                "magnetic bottle holder for a bicycle with auto locking",
                "a method of manufacturing ceramic tiles",
            ],
        )
        assert scores[0] > scores[1], f"relevant doc should win: {scores}"

    def test_max_clause_rerank_end_to_end(self, reranker, sample_user_idea, sample_patent_candidates):
        out = reranker.max_clause_rerank(sample_user_idea, sample_patent_candidates, top_k=2)
        assert out
        assert out[0]["max_similarity_score"] >= out[-1]["max_similarity_score"]
        assert out[0]["matched_limitation"]


# ==================================================================
# 8. LLM tests (require Ollama)
# ==================================================================
@pytest.mark.llm
class TestOllamaStructuredOutput:
    def test_ollama_available(self, ollama_available):
        if not ollama_available:
            pytest.skip("Ollama not running")
        from langgraph_state_machine import get_available_ollama_models
        assert get_available_ollama_models(), "no Ollama models installed"

    def test_structured_extraction_returns_schema(self, ollama_available):
        if not ollama_available:
            pytest.skip("Ollama not running")

        from input_analysis import PatentQueryAnalysis
        from langgraph_state_machine import call_structured_llm

        out = call_structured_llm(
            PatentQueryAnalysis,
            "Extract patent search features.",
            "A magnetic water bottle holder for bicycles that locks automatically.",
        )
        assert isinstance(out, PatentQueryAnalysis)
        assert out.summary
        assert out.technical_mechanisms

    def test_full_langgraph_run(self, ollama_available, sample_reranked_patents):
        if not ollama_available:
            pytest.skip("Ollama not running")

        from langgraph_state_machine import patent_synthesis_pipeline

        state = {
            "user_idea": "magnetic bottle holder for bicycles",
            "reranked_patents": sample_reranked_patents,
            "deconstructed_idea": [],
            "deconstructed_patents": [],
            "diff_matrix": {},
            "obviousness_analysis": {},
            "final_report": {},
        }
        out = patent_synthesis_pipeline.invoke(state)
        assert out["final_report"]["novelty_status"] in {
            "Anticipated (102)",
            "Obvious Combination (103)",
            "Potentially Novel",
        }
        assert out["final_report"]["legal_disclaimer"]


# ==================================================================
# 9. Network tests (Google Patents)
# ==================================================================
@pytest.mark.network
class TestPriorArtNetwork:
    def test_google_patents_returns_something(self, network_available):
        if not network_available:
            pytest.skip("no network")

        from input_analysis import PatentQueryAnalysis
        from prior_evidence_retrieval import AsyncPatentRetriever

        analysis = PatentQueryAnalysis(
            summary="magnetic bottle holder",
            technical_mechanisms=["magnetic", "latching"],
            legal_synonyms=["receptacle", "vehicle"],
            search_query="magnetic latching receptacle vehicle",
        )
        retriever = AsyncPatentRetriever()
        docs = asyncio.run(retriever.execute_parallel_retrieval(analysis, ["B62J", "H01F"]))

        # NOTE: Google's public endpoint is flaky. Empty is acceptable here;
        # we only assert the shape when records come back.
        for d in docs:
            assert d.doc_id
            assert d.status_tag in {"Expired (Prior Art Only)", "Active Patent", "Status Unknown"}
        print(f"\n[network] retrieved {len(docs)} prior-art docs")


# ==================================================================
# 10. Cross-module wiring
# ==================================================================
@pytest.mark.unit
class TestCrossModuleWiring:
    def test_claim_parser_output_feeds_reranker(self, sample_patent_text, monkeypatch):
        """End-to-end: text -> independent claims -> limitations -> reranker scores."""
        from claim_parser import extract_independent_claims, parse_claim_limitations
        import cross_encoder_reranker as cer

        claims = extract_independent_claims(sample_patent_text)
        assert claims
        all_lims: List[str] = []
        for c in claims:
            all_lims.extend(parse_claim_limitations(c["raw_text"])["limitations"])
        assert all_lims

        instance = cer.PatentReranker.__new__(cer.PatentReranker)
        instance.score_pairs = lambda q, t: [0.5] * len(t)  # type: ignore[assignment]
        out = instance.max_clause_rerank(
            "magnetic bottle holder",
            [{"id": "X", "title": "T", "text": sample_patent_text}],
            top_k=1,
        )
        assert out and out[0]["all_parsed_limitations"]

    def test_input_analysis_schema_roundtrips_into_retrieval(self):
        from input_analysis import PatentQueryAnalysis
        from prior_evidence_retrieval import GooglePatentsQueryBuilder

        analysis = PatentQueryAnalysis(
            summary="magnetic bottle holder",
            technical_mechanisms=["magnetic"],
            legal_synonyms=["receptacle"],
            search_query="magnetic receptacle",
        )
        q = GooglePatentsQueryBuilder.build_query_c_broad_boolean(analysis)
        assert "magnetic bottle holder" in q
        assert "receptacle" in q