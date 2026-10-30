"""원문 수치를 상수가 그대로 내는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import llmd_figures as m


def test_nemotron_deltas_match_article():
    for label, old, new, stated in m.NEMOTRON:
        assert abs(m.change_pct(old, new) - stated) < 1, label   # 원문 Δ 는 정수로 내린 값


def test_nemotron_is_prefill_heavy():
    rows = {label: (old, new) for label, old, new, _ in m.NEMOTRON}
    for i in (0, 1):
        assert m.input_share(rows["Input tok/s"][i], rows["Total tok/s"][i]) > 98


def test_claims_order_and_conversion():
    hi = [h for *_, h, _ in m.CLAIMS]
    assert hi == sorted(hi, reverse=True)
    assert m.CLAIMS[0][2] == 13.9
    assert round(m.CLAIMS[4][2], 2) == 1.67                  # 40% 감소 = 1/0.6 배


def test_wide_ep_per_gpu_implies_16_gpus():
    cluster, per_gpu = m.WIDE_EP
    assert round(cluster / per_gpu) == 16


@pytest.mark.parametrize("name,fn", m.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
