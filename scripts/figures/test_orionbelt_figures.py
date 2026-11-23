"""검사 수와 엔진 기준이 원문 값인지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import orionbelt_figures as m


def test_skos_checks_match_readme():
    sizes = [len(ids) for _, _, ids in m.SKOS_TIERS]
    assert sizes == [9, 9, 5] and sum(sizes) == 23           # README: 23 checks in three tiers
    every = [i for _, _, ids in m.SKOS_TIERS for i in ids]
    assert len(set(every)) == 23
    assert len(m.AUTOFIX) == 7 and m.AUTOFIX <= set(every)    # README: 7 of the checks can be repaired
    fixed = [sum(i in m.AUTOFIX for i in ids) for _, _, ids in m.SKOS_TIERS]
    assert fixed == [3, 4, 0]


def test_quality_rules_match_registry():
    assert sum(n for _, n, _ in m.QUALITY) == 22
    assert sum(on for _, _, on in m.QUALITY) == 14
    assert all(0 < on <= n for _, n, on in m.QUALITY)


def test_sparql_engine_constants():
    assert m.OXIGRAPH_MIN_TRIPLES == 20_000
    assert m.EVEN_AT < m.OXIGRAPH_MIN_TRIPLES < m.FAST_AT
    assert m.DEFAULT_TIMEOUT_SECONDS == 10.0 and m.DEFAULT_MAX_ROWS == 1000


@pytest.mark.parametrize("name,fn", m.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
