"""옮겨 적은 수치에서 계산한 배율이 논문 본문 수치와 맞는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import ptq_figures as p


def test_benchmark_drop_matches_paper():
    assert p.bench_mean_drop() == pytest.approx(-0.43, abs=0.005)  # 본문 "0.43 percentage points"


def test_hidden_error_ratios_match_paper():
    r = {k: a / b for k, (a, b) in p.HIDDEN_ERR.items()}
    assert r["Qwen3-32B 절대 오차"] == pytest.approx(5.5, abs=0.05)
    assert r["Qwen3-32B 상대 오차"] == pytest.approx(6.7, abs=0.05)
    assert r["OLMo3-7B 절대 오차"] == pytest.approx(20.2, abs=0.05)
    assert r["OLMo3-7B 상대 오차"] == pytest.approx(3.7, abs=0.05)


def test_rotation_attenuation_matches_paper():
    assert p.attenuation_measured() == pytest.approx(78.6, abs=0.3)
    assert 1 / p.mu_d(p.HIDDEN_DIM) == pytest.approx(89.7, abs=0.05)


def test_intervention_ratios_match_paper():
    base, rem, rev = (r[1] for r in p.INTERVENE)
    assert rem / base == pytest.approx(2.94, abs=0.01)
    assert rev / base == pytest.approx(8.41, abs=0.01)


def test_cancellation_total():
    assert p.CANCEL_INTER + p.CANCEL_ALIGN == pytest.approx(81.8)


@pytest.mark.parametrize("name", list(p.FIGURES))
def test_svgs_are_well_formed(name):
    ET.fromstring(p.FIGURES[name]())
