"""옮겨 적은 표 수치가 논문의 주장과 맞는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import skipkv_figures as s


def test_throughput_claims():
    sk, rk, fk = s.THROUGHPUT["SkipKV"], s.THROUGHPUT["R-KV"], s.THROUGHPUT["FullKV"]
    # 논문의 9.6배·7.6배는 FullKV 배치 50 값 기준
    assert round(max(sk) / fk[1], 1) == 9.6
    assert round(max(rk) / fk[1], 1) == 7.6
    # 각자 최고점끼리는 약 6.2배
    assert round(max(sk) / max(v for v in fk if v), 1) == 6.2
    # 같은 배치 10 에서 R-KV 대비 1.7배
    assert round(sk[0] / rk[0], 1) == 1.7


def test_ablation_deltas():
    base = s.ABLATION[0][1]
    final = s.ABLATION[-1][1]
    assert [round(f - b, 1) for f, b in zip(final, base)] == [13.3, 20.0, 13.3]
    grouping_gain = [round(f - p, 1) for f, p in zip(final, s.ABLATION[-2][1])]
    assert grouping_gain == [10.0, 10.0, 10.0]


def test_padding_constant():
    assert {b - v0 for b, v0, _, _, _ in s.GROUPING} == {138}
    assert {b - v1 for b, _, v1, _, _ in s.GROUPING} == {9}


@pytest.mark.parametrize("name,fn", s.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
