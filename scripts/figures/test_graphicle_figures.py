"""옮겨 적은 벤치마크 수치가 원문과 맞는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import graphicle_figures as g


def test_retrieval_matches_source():
    rows = {name: vals for name, vals, _ in g.RETRIEVAL}
    assert rows["Graphicle 0.1.8"] == (43, 75, 81)
    assert rows["BM25 (SWE-bench 방식)"] == (35, 61, 77)
    # 0.1.2 의 랭킹은 grep 보다 낮았다
    assert all(a < b for a, b in zip(rows["Graphicle 0.1.2"], rows["grep (이슈의 이름)"]))


def test_pack_recall_shape():
    # 대회 후 pack 이 모든 예산에서 대회 전 이상, 자른 저장소는 3% 미만
    assert all(a >= b for a, b in zip(g.PACK_AFTER, g.PACK_BEFORE))
    assert max(g.TRUNCATION) < 3
    assert g.PACK_AFTER[0] == 58.1 and g.PACK_AFTER[-1] == 77.9


def test_cache_tradeoff():
    costs = {n: c for n, c, _ in g.CACHE_1M}
    recall = {n: r for n, _, r in g.CACHE_1M}
    # 앞부분 비율을 올릴수록 후속 비용은 내려간다
    order = ["standard", "cache 50%", "cache 70%", "cache 90%", "cache 95%", "stable"]
    assert all(costs[a] > costs[b] for a, b in zip(order, order[1:]))
    # 90% 는 standard 비용의 5분의 1 수준이면서 포함 비율 90% 이상
    assert costs["cache 90%"] < costs["standard"] / 4.5 and recall["cache 90%"] > 90
    assert recall["stable"] < 20


@pytest.mark.parametrize("name,fn", g.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
