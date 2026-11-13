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
