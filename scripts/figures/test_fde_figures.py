"""옮겨 적은 집계(편수 · 단어 · 발행일 · 지연 예산)가 원문과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import fde_figures as f


def test_totals_match_readme_count():
    assert sum(n for _, n, _, _ in f.STEPS.values()) == 140
    assert sum(v for _, _, v, _ in f.STEPS.values()) == 10
    assert sum(w for *_, w in f.STEPS.values()) == 533224


def test_pub_lists_match_step_counts():
    for st, (_, n, _, _) in f.STEPS.items():
        assert len(f.PUB[st]) == n


def test_every_step_belongs_to_one_phase():
    flat = [s for _, steps, _ in f.PHASES for s in steps]
    assert sorted(flat) == list(f.STEPS)


def test_prove_value_is_thinnest_and_voice_is_largest():
    n = {st: v[1] for st, v in f.STEPS.items()}
    assert n[8] == n[9] == min(n.values()) == 4
    assert max(n, key=n.get) == 13


def test_publish_date_claims():
    all_d = [f.d(s) for v in f.PUB.values() for s in v]
    assert sum(1 for x in all_d if x >= f.REPO_CREATED) == 15
    assert sum(1 for x in all_d if x.year < 2026) == 4
    step8 = [f.d(s) for s in f.PUB[8]]
    assert (max(step8) - min(step8)).days == 6
    late13 = [f.d(s) for s in f.PUB[13] if f.d(s) >= f.date(2026, 10, 19)]
    assert len(late13) == 20


def test_voice_budget_sums_to_target_and_llm_overrun():
    assert sum(b for _, b, _ in f.VOICE) == f.VOICE_TARGET
    llm = dict((n, (b, m)) for n, b, m in f.VOICE)["LLM 첫 토큰"]
    assert llm[1] - llm[0] == 220


@pytest.mark.parametrize("name,fn", f.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
