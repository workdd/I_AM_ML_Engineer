"""원문 수치를 상수와 모형이 그대로 내는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import pgjev_figures as p


def test_token_model_matches_article():
    assert p.tokens_per_row(1) == 435                       # 1행 단독 약 435
    assert round(p.tokens_per_row(20)) in range(170, 181)   # 20행 묶음 행당 약 175
    assert round((p.tokens_per_row(20) / p.tokens_per_row(40) - 1) * 100) == 4   # 20행이 40행보다 4% 많다


def test_accuracy_falls_with_batch_size():
    lows = [lo for _, _, (lo, _) in p.ACCURACY]
    assert lows == sorted(lows, reverse=True)
    assert p.ACCURACY[0][2] == (100, 100)


def test_versions():
    rows = {label: (old, new) for label, old, new in p.TIMINGS}
    assert rows["전체 스캔, 새 조건"] == (8.5, 3.5)
    assert rows["LIMIT 3, 새 조건"] == (8.4, 0.6)
    old_t, new_t = p.TOKENS
    assert round((1 - new_t / old_t) * 100) == 12          # CHANGELOG "12 % fewer input tokens"
    assert round(new_t * 0.042 / 1e6, 3) == 0.012          # 약 $0.012


@pytest.mark.parametrize("name,fn", p.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
