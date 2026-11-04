"""옮겨 적은 수가 레포 문서와 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import egm_figures as e


def test_stage_order_is_five_to_nine():
    assert [s for s, *_ in e.REVIEW_STAGES] == ["5", "6", "7", "8", "9"]


def test_review_has_nine_reviewers_and_total_matches_record():
    # egm-review Uninterrupted Full Flow: 1 + 1 + 3 + 1 + 3
    assert [len(r) for *_, r in e.REVIEW_STAGES] == [1, 1, 3, 1, 3]
    assert e.review_total() == 9
    # TESTING.md: Codex · Muse "15 native reviewers" (brainstorm 6 + stage 9)
    assert e.BRAINSTORM_LENSES + e.review_total() == e.RECORDED_TOTAL == 15


def test_zoom_out_runs_twice_and_style_in_both_critic_and_readiness():
    names = [rv for *_, r in e.REVIEW_STAGES for rv in r]
    assert names.count("zoom-out") == 2
    assert names.count("style") == 2


def test_harness_rows_and_codes():
    assert len(e.HARNESS) == 8
    assert all(len(row) == 7 and set(row[3:]) <= set(e.STATUS) for row in e.HARNESS)
    assert all(row[3] == "P" for row in e.HARNESS)  # 설치는 모두 Pass
    assert sum(row[6] in "PA" for row in e.HARNESS) == 5  # Review 까지 Pass 는 5곳


@pytest.mark.parametrize("name,fn", e.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
