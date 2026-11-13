"""원본 그림의 구획과 수치가 한국어판에 빠짐없이 옮겨졌는지 확인한다."""
import re

import llmwiki_arch_ko as a


def test_sections_match_original_counts():
    # 원본: 3층 · 3대 작업 · 인덱스 2 · 역할 2 · 흐름 3+1 · 산출 4 · 활용 8
    assert (len(a.LAYERS), len(a.OPS), len(a.INDEX), len(a.ROLES)) == (3, 3, 2, 2)
    assert (len(a.FLOW), len(a.OUTPUTS), len(a.USES)) == (3, 4, 8)


def test_numbers_copied_from_original():
    text = a.page()
    assert "10~15" in text and "15개" in text


def test_page_is_korean():
    text = re.sub(r"<[^>]+>", "", a.page())
    assert not re.search(r"[一-鿿]", text)  # 한자 없음


def test_arrows_match_original():
    # 원본: 층 사이 화살표 2개, 수집 흐름 직렬 화살표 3개 + 요점 논의에서 산출물 4개로 갈라지는 화살표, 역할 사이 협업 연결선
    svg = a.flow_svg()
    assert svg.count('marker-end="url(#ah)"') == 3
    assert svg.count('marker-end="url(#ahg)"') == len(a.OUTPUTS) == 4
    page = a.page()
    assert page.count("<div class='arrow'>") == len(a.LAYERS) - 1
    assert "협업</span><svg" in page
