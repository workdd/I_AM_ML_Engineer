"""그림 생성기의 옮겨 적은 수치와 파생 계산이 논문 본문 수치와 맞는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import graphrag_figures as g


def test_ragsearch_gaps_match_paper():
    gaps = g.rs_gaps()
    assert gaps["단일 검색"] == pytest.approx(27.23, abs=0.01)  # 5.2.1절
    # 5.2.2절 "26.59"는 두 학습 없는 에이전트 격차의 평균
    assert (gaps["Search-o1"] + gaps["GraphSearch"]) / 2 == pytest.approx(26.59, abs=0.01)


def test_pathrag_averages_match_paper():
    assert g.pr_avg(g.PR_T1["MS GraphRAG"]) == pytest.approx(59.93, abs=0.01)
    assert g.pr_avg(g.PR_T1["LightRAG"]) == pytest.approx(57.09, abs=0.01)
    assert g.pr_avg(g.PR_ABL["무작위 정렬 대비"]) == pytest.approx(56.44, abs=0.01)
    assert g.pr_avg(g.PR_ABL["홉 수 우선 정렬 대비"]) == pytest.approx(55.64, abs=0.01)
    assert g.pr_avg(g.PR_ABL["평평한 프롬프트 대비"]) == pytest.approx(55.19, abs=0.01)
    t6 = g.PR_T6
    assert 1 - t6["PathRAG"] / t6["LightRAG"] == pytest.approx(0.1369, abs=1e-4)
    assert 1 - t6["PathRAG-lt"] / t6["LightRAG"] == pytest.approx(0.4041, abs=1e-4)


def test_table_shapes():
    for rows in (g.RS_T1, *g.RS_T2.values()):
        assert all(len(r) == len(g.DATASETS) for r in rows.values())
    for table in (*g.PR_T1.values(), *g.PR_ABL.values()):
        assert len(table) == len(g.PR_DIMS)
        assert all(len(r) == len(g.PR_DATASETS) for r in table)


def test_flow_example_prefers_unbranched_path():
    paths = g.candidate_paths(g.FLOW_EDGES, "A", "T")
    assert sorted(map(tuple, paths)) == [("A", "B", "C", "T"), ("A", "H", "D", "T"), ("A", "H", "T")]
    w, _, scores = g.flow_weights(paths, "A", "T")
    assert w[("A", "H")] == pytest.approx(0.5)
    assert w[("H", "T")] == pytest.approx(0.2)
    assert ("D", "T") not in w  # H→D 가 θ 이하라 전파 중단
    assert max(scores, key=scores.get) == ("A", "B", "C", "T")


@pytest.mark.parametrize("name", list(g.FIGURES))
def test_svgs_are_well_formed(name):
    ET.fromstring(g.FIGURES[name]())


def test_graphragbench_tables_consistent():
    for ds in ("Novel", "Medical"):
        assert set(g.GB_ACC[ds]) == set(g.GB_RET[ds])
        assert all(len(v) == len(g.GB_TASKS) for v in g.GB_ACC[ds].values())
    # 부록 G.4 본문 평균값
    assert sum([57.56, 56.01, 61.95, 60.91]) / 4 == pytest.approx(59.10, abs=0.01)


def test_graphragbench_scale_row_copies_medical():
    # Table 17 의 소설 RAG 56k 행이 의료 Table 9 rerank RAG 값과 같다는 노트의 지적
    assert g.GB_SCALE["RAG"][0] == g.GB_ACC["Medical"][g.GB_RAG]
    assert g.GB_SCALE["HippoRAG2"][0] == g.GB_ACC["Novel"]["HippoRAG2"]


def test_graphragbench_level_counts():
    assert g.gb_beats_rag("Novel") == [0, 8, 7, 9]
    assert g.gb_beats_rag("Medical") == [1, 3, 1, 7]
