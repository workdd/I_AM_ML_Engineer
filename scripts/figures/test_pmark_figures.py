"""판정 규칙이 원문 문장과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import pmark_figures as s


def res(project="재무", file_m=(), data_m=()):
    return ("r", project, set(file_m), set(data_m), "")


def test_markings_are_conjunctive():
    # 원문: markings are conjunctive (boolean AND)
    assert s.decide({"재무"}, {"PII"}, res(file_m={"PII", "Raw Data"})) == s.HIDDEN
    assert s.decide({"재무"}, {"PII", "Raw Data"}, res(file_m={"PII", "Raw Data"})) == s.READ


def test_role_does_not_override_marking():
    # 원문: Regardless of role, a user cannot access a file unless all marking requirements are met
    assert s.decide({"재무"}, set(), res(file_m={"PII"})) == s.HIDDEN


def test_eligibility_does_not_provision_access():
    # 원문: Markings should not be used to provision access
    assert s.decide(set(), {"PII"}, res(file_m={"PII"})) == s.NOROLE


def test_data_marking_shows_metadata_only():
    # 원문: file 요건 충족 + 상속된 data 요건 미충족이면 존재와 메타데이터만 보인다
    assert s.decide({"재무"}, set(), res(data_m={"PII"})) == s.META


def test_matrix_uses_all_four_outcomes():
    cells = {c for row in s.matrix() for c in row}
    assert cells == {s.HIDDEN, s.NOROLE, s.META, s.READ}


@pytest.mark.parametrize("name,fn", s.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
