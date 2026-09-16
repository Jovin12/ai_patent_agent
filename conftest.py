"""
Shared fixtures and import-safety shims.

Several of your modules do work at import time (loading spacy models,
opening Ollama connections, initializing ChromaDB). These shims let the
unit tests import the modules without requiring those services to be up.
"""
import os
import sys
import types
from typing import Any, Dict, List

import pytest


# ------------------------------------------------------------------
# 1. Neutralize heavy import-time side effects BEFORE test modules import
# ------------------------------------------------------------------
# claim_parser.py loads spacy at import — stub it out for unit tests.
if "spacy" not in sys.modules:
    spacy_stub = types.ModuleType("spacy")

    class _FakeDoc:
        def __init__(self, text: str):
            self.text = text

    def _fake_load(_name: str):
        return lambda text: _FakeDoc(text)

    spacy_stub.load = _fake_load
    sys.modules["spacy"] = spacy_stub


# ------------------------------------------------------------------
# 2. Sample data fixtures
# ------------------------------------------------------------------
@pytest.fixture(scope="session")
def sample_patent_text() -> str:
    """A small synthetic patent with 3 independent claims and 2 dependent ones."""
    return """1. A magnetic bottle holder for a bicycle, comprising:
   a mounting base configured to attach to a bicycle frame;
   a receptacle defining a cavity sized to receive a bottle;
   a magnetic alignment element disposed on the receptacle; and
   a latching mechanism configured to lock the bottle in the receptacle
   upon magnetic alignment.

2. The holder of claim 1, wherein the latching mechanism is spring-loaded.

3. The holder of claim 1, further comprising a release lever.

4. A method of securing a bottle to a bicycle, comprising:
   providing a receptacle with a magnetic alignment element;
   inserting the bottle into the receptacle; and
   automatically locking the bottle in response to magnetic alignment.

5. The method of claim 4, wherein the locking step uses a mechanical latch.

6. A bicycle assembly, consisting of:
   a bicycle frame;
   a bottle holder according to claim 1; and
   a bottle having a ferromagnetic portion.
"""


@pytest.fixture(scope="session")
def sample_claim_block() -> str:
    return (
        "A magnetic bottle holder for a bicycle, comprising: "
        "a mounting base configured to attach to a bicycle frame; "
        "a receptacle defining a cavity sized to receive a bottle; "
        "a magnetic alignment element disposed on the receptacle; and "
        "a latching mechanism configured to lock the bottle upon magnetic alignment."
    )


@pytest.fixture(scope="session")
def sample_user_idea() -> str:
    return "A magnetic water bottle holder for bicycles that locks automatically upon alignment"


@pytest.fixture(scope="session")
def sample_reranked_patents() -> List[Dict[str, Any]]:
    return [
        {
            "patent_id": "KR102370398B1",
            "title": "Closure device for attaching container to bicycle",
            "matched_limitation": (
                "a receiving latch configured to mechanically lock a bottle "
                "upon magnetic alignment"
            ),
        },
        {
            "patent_id": "US20210169205A1",
            "title": "Device Having Beverage Container Holder",
            "matched_limitation": (
                "a receiving device for detachably fixing the beverage container "
                "in place using a mechanical release"
            ),
        },
    ]


@pytest.fixture(scope="session")
def sample_patent_candidates() -> List[Dict[str, Any]]:
    """Candidates shaped the way PatentReranker.max_clause_rerank expects."""
    return [
        {
            "id": "US1234567A",
            "title": "Magnetic Bottle Holder",
            "text": (
                "1. A magnetic bottle holder for a bicycle, comprising: "
                "a mounting base; a receptacle defining a cavity; "
                "a magnetic alignment element; and a latching mechanism."
            ),
        },
        {
            "id": "US7654321B2",
            "title": "Beverage Container Retainer",
            "text": (
                "1. A retainer for a beverage container, comprising: "
                "a base; a cradle; and a mechanical release mechanism."
            ),
        },
    ]


# ------------------------------------------------------------------
# 3. Optional service availability markers
# ------------------------------------------------------------------
def _ollama_up() -> bool:
    try:
        import urllib.request
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return r.status == 200
    except Exception:
        return False


def _network_up() -> bool:
    try:
        import socket
        socket.create_connection(("1.1.1.1", 80), timeout=2).close()
        return True
    except Exception:
        return False


def _torch_cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:
        return False


@pytest.fixture(scope="session")
def ollama_available() -> bool:
    return _ollama_up()


@pytest.fixture(scope="session")
def network_available() -> bool:
    return _network_up()


@pytest.fixture(scope="session")
def cuda_available() -> bool:
    return _torch_cuda()


# Skip helpers used by test module
def pytest_collection_modifyitems(config, items):
    import pytest as _pytest

    need_ollama = not _ollama_up()
    need_net = not _network_up()
    need_gpu = not _torch_cuda()

    for item in items:
        if "llm" in item.keywords and need_ollama:
            item.add_marker(_pytest.mark.skip(reason="Ollama not running on :11434"))
        if "network" in item.keywords and need_net:
            item.add_marker(_pytest.mark.skip(reason="No network connectivity"))
        if "gpu" in item.keywords and need_gpu:
            item.add_marker(_pytest.mark.skip(reason="CUDA not available"))