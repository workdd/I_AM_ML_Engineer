"""옮겨 적은 구성이 원문·저장소와 맞는지, SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import deepinsight_figures as d


def test_graph_edges_match_builder():
    # 수정 엣지는 plan_reviewer → planner 하나뿐, supervisor 로 가는 길은 plan_reviewer 를 거친다
    back = [(a, b) for a, b, _ in d.GRAPH_EDGES if d.GRAPH_NODES.index(b) < d.GRAPH_NODES.index(a)]
    assert back == [("plan_reviewer", "planner")]
    assert ("plan_reviewer", "supervisor", "should_proceed_to_supervisor") in d.GRAPH_EDGES
    assert d.MAX_NODE_EXECUTIONS == 25 and d.MAX_PLAN_REVISIONS == 10


def test_supervisor_order_tracks_after_every_agent():
    names = [n for n, _ in d.SUPERVISOR_ORDER]
    assert names == ["Coder", "Tracker", "Validator", "Tracker", "Reporter", "Tracker"]


def test_model_table():
    rows = {a: (m, th, c) for a, m, th, c in d.MODELS}
    assert len(rows) == 7
    assert [a for a, (_, th, _) in rows.items() if th] == ["Planner"]
    assert sorted(a for a, (_, _, c) in rows.items() if c) == ["Coder", "Reporter", "Supervisor"]
    assert rows["Coordinator"][0] == "Haiku" and rows["Planner"][0] == "Opus"


@pytest.mark.parametrize("name,fn", d.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
