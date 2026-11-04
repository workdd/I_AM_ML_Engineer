"""옮겨 적은 수가 평가 보고서와 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import carrel_figures as c


def test_cross_document_graph_gain_is_4_52pp_and_top10_drops():
    cov = dict((n, v) for n, _, v in c.ABLATION)["교차 문서 coverage"]
    top10 = dict((n, v) for n, _, v in c.ABLATION)["교차 문서 top10 완전"]
    assert round(cov[0] - cov[1], 2) == 4.52
    assert top10[0] < top10[1]


def test_overall_graph_does_not_improve_hit5_or_mrr():
    hit5 = c.ABLATION[0][2]
    assert round(hit5[0] - hit5[1], 2) == -1.09
    assert c.MRR[0] < c.MRR[1] < c.MRR[2]


def test_question_counts_split_into_single_and_cross():
    n = {name: k for name, k, _ in c.ABLATION}
    assert n["단일 문서 coverage"] + n["교차 문서 coverage"] == 275


def test_answer_completeness_matches_report():
    rows = {name: (ks, n) for name, ks, n in c.COMPLETENESS}
    ks, n = rows["실문서 교차"]
    assert [round(c.pct(k, n)) for k in ks] == [60, 76, 92]
    assert rows["실문서 단일"][0] == (29, 29, 29)
    # 전체 75 = 실문서 55 (30 + 25) + 합성 20
    assert rows["실문서 단일"][1] + rows["실문서 교차"][1] + rows["합성"][1] == rows["전체"][1] == 75
    for i in range(3):
        assert sum(rows[g][0][i] for g in ("실문서 단일", "실문서 교차", "합성")) == rows["전체"][0][i]


def test_skill_costs_more_than_single_retrieval():
    lat, tok = c.COST["latency"], c.COST["tokens"]
    assert round(lat[1] - lat[0], 1) == 11.9
    assert round((tok[1] - tok[0]) / 1000, 1) == 35.1


def test_rebuild_order_entities_more_stable_than_relations():
    rows = {name: (a, e) for name, a, e in c.REBUILD}
    for base in (0, 1):
        assert all(rows[n][base][0] == 100 for n in rows)  # cached rebuild 동일
        assert rows["엔티티"][base][1] > rows["관계"][base][1]
        assert rows["fact 출처 청크"][base][2] > 97
        for n in rows:  # labels 재추출이 고정보다 덜 겹친다
            assert rows[n][base][2] <= rows[n][base][1]


@pytest.mark.parametrize("name,fn", c.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
