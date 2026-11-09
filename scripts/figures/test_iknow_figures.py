"""옮겨 센 값이 write-up 의 총계 문장과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import iknow_figures as k


def test_predicate_rows_sum_to_278_and_mechanism_totals():
    assert sum(sum(r[1:]) for r in k.PREDICATES) == k.N_PREDICATES
    assert [sum(r[i] for r in k.PREDICATES) for i in (1, 2, 3)] == [161, 24, 93]


def test_model_kinds_match_model_row():
    model = [r for r in k.PREDICATES if r[0].startswith("모델")][0]
    assert sum(v for _, v in k.MODEL_KINDS) == sum(model[1:]) == 28


def test_copied_is_the_majority():
    copied = sum(k.PREDICATES[0][1:])
    assert copied == 161 and copied > k.N_PREDICATES / 2


def test_class_rows_sum_to_181_and_write_up_counts():
    assert sum(sum(r[1:]) for r in k.CLASSES) == k.N_CLASSES
    assert [sum(r[i] for r in k.CLASSES) for i in (1, 2, 3)] == [90, 16, 75]
    by = {r[0]: sum(r[1:]) for r in k.CLASSES}
    assert by["edge (관계 타입)"] == 46                              # "the forty-six relationship types"
    assert by["activity"] + by["activity type"] == 57    # "Fifty-seven of the 181 describe an activity"


@pytest.mark.parametrize("name,fn", k.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
