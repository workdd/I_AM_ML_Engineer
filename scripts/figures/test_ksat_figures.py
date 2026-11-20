"""옮겨 적은 수치가 모델 카드 주장과 맞는지, 생성한 SVG 가 깨지지 않았는지 확인한다."""
import xml.etree.ElementTree as ET

import pytest

import ksat_figures as k


def scores(name_prefix):
    return next(v for n, _, v in k.SCORES if n.startswith(name_prefix))


def test_only_stage1_has_perfect_majority():
    assert [m for _, m in scores("1차")] == [100, 100, 100]
    assert [m for _, m in scores("원본")] == [96, 96, 96]
    assert [m for _, m in scores("2차")] == [96, 96, 96]


def test_stage2_mean_gain_range():
    # 모델 카드: 2차 step 10 은 원본보다 선택과목별 평균 1.5~2.6점 높다
    gains = [round(b[0] - a[0], 1) for a, b in zip(scores("원본"), scores("2차"))]
    assert min(gains) == 1.5 and max(gains) == 2.6


def test_stage1_mean_is_highest():
    for i in range(3):
        assert scores("1차")[i][0] >= max(scores("원본")[i][0], scores("2차")[i][0])


def test_stage1_composition():
    assert sum(n for _, n in k.STAGE1) == k.STAGE1_TOTAL == 683


def test_stage2_rows_and_mismatch():
    assert k.STAGE2_UNIQUE + k.STAGE2_DOUBLED == k.STAGE2_ROWS == 1145
    # 구성 항목 합은 1,082로 고유 1,078과 4 차이 (원문 수치 그대로 둔다)
    assert sum(n for _, n in k.STAGE2) - k.STAGE2_UNIQUE == 4


@pytest.mark.parametrize("name,fn", k.FIGS)
def test_svg_is_well_formed(name, fn):
    root = ET.fromstring(fn())
    assert root.get("viewBox")
    assert root.find("{http://www.w3.org/2000/svg}title").text
