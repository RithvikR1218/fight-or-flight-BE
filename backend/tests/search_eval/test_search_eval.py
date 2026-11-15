"""Judged search queries on the real data files (no DB). The lexical variant runs in every
`pytest`; the semantic variant runs when SEARCH_EVAL_VECTORS points at an .npz of place vectors
(python -m scripts.load_places --offline-embed PATH.npz) and fastembed can embed queries."""
import os

import pytest

from .cases import CASES
from .harness import Runner, build_index, eval_vectors_path

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "usc_google_maps_places.geojson")
pytestmark = [pytest.mark.search_eval,
              pytest.mark.skipif(not os.path.exists(DATA), reason="data/usc_google_maps_places.geojson missing")]

# Judged queries the ranker still gets wrong, with the reason (kept visible, not deleted).
KNOWN_FAILURES_LEXICAL = {
}
KNOWN_FAILURES_SEMANTIC = {   # measured with BAAI/bge-small-en-v1.5
    "cheap eats": "semantic pulls Henry's Market (a grocery) into the top 5; passes lexical-only",
    "watch the game": "held-out: only 2 results (Tom's Watch Bar by name + a far 'Sports Bar'): a name match "
                      "makes the lexical score high, which shrinks the semantic weight",
    "spicy food": "held-out: places NAMED 'Spicy ...' score 1.0 lexically, so ordinary restaurants fall under "
                  "the relevance gate (2 results)",
    "hangover food": "held-out: the embedding matches bars / night spots and a liquor store",
    "cold brew": "held-out: names containing 'Brew' (breweries) beat cafes; 'cold brew' is in no type label",
}


@pytest.fixture(scope="module")
def lexical_runner():
    return Runner(build_index())


@pytest.fixture(scope="module")
def semantic_runner():
    path = eval_vectors_path()
    if not path:
        pytest.skip("SEARCH_EVAL_VECTORS not set: semantic eval skipped")
    from app.services.search_semantic import QueryEncoder

    enc = QueryEncoder(mode="on")
    if enc.disabled_reason or enc.encode("coffee") is None:
        pytest.skip(f"query embedding unavailable: {enc.disabled_reason}")
    return Runner(build_index(path), encoder=enc)


def _ids(cases):
    return [c.q or f"<{','.join(c.categories or [])}>" for c in cases]


def _run(runner, case, known, held_out_xfail=False):
    if case.q in known:
        pytest.xfail(known[case.q])
    if held_out_xfail and case.held_out:
        _, fails = runner.check(case)
        if fails:
            pytest.xfail("held-out paraphrase (a measurement, not a gate): " + "; ".join(fails))
        return
    _, fails = runner.check(case)
    assert not fails, f"{case.q!r}: " + "; ".join(fails)


@pytest.mark.parametrize("case", [c for c in CASES if not c.semantic_only], ids=_ids([c for c in CASES if not c.semantic_only]))
def test_lexical(lexical_runner, case):
    # held-out paraphrases mostly need meaning (the semantic signal): reported, not required, here
    _run(lexical_runner, case, KNOWN_FAILURES_LEXICAL, held_out_xfail=True)


@pytest.mark.parametrize("case", CASES, ids=_ids(CASES))
def test_semantic(semantic_runner, case):
    _run(semantic_runner, case, KNOWN_FAILURES_SEMANTIC)
