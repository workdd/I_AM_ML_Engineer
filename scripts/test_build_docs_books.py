"""위키독스 형식 책(books/<책>/TOC.md)을 사이트 목차로 옮기는 부분을 확인한다."""
from pathlib import Path

import build_docs as b


def test_parse_toc_depth_and_paths(tmp_path: Path):
    toc = tmp_path / "TOC.md"
    toc.write_text("# 목차\n\n- [01. 시작](pages/01.md)\n  - [01-1. 설치](pages/01-1.md)\n- [02. 끝](pages/02.md)\n",
                   encoding="utf-8")
    assert b.parse_toc(toc) == [(0, "01. 시작", "pages/01.md"), (1, "01-1. 설치", "pages/01-1.md"),
                                (0, "02. 끝", "pages/02.md")]


def test_every_book_page_in_toc_exists():
    for book in b.BOOKS.iterdir():
        if (book / "TOC.md").exists():
            for _, _, rel in b.parse_toc(book / "TOC.md"):
                assert (book / rel).exists(), f"{book.name}: TOC 에 있는데 파일 없음 {rel}"


def test_book_pages_have_no_h1():
    # 위키독스 GitHub 연동 책 규칙: 페이지 제목은 TOC.md 가 정한다
    for page in b.BOOKS.glob("*/pages/*.md"):
        first = page.read_text(encoding="utf-8").lstrip().splitlines()[0]
        assert not first.startswith("# "), f"{page} 첫 줄이 H1"


def test_draft_book_is_not_copied(tmp_path, monkeypatch):
    books = tmp_path / "books"
    for name, draft in (("open", False), ("hidden", True)):
        (books / name / "pages").mkdir(parents=True)
        (books / name / "README.md").write_text(f"# {name}\n", encoding="utf-8")
        (books / name / "TOC.md").write_text("# 목차\n\n- [01. a](pages/01.md)\n", encoding="utf-8")
        (books / name / "pages" / "01.md").write_text("본문\n", encoding="utf-8")
        if draft:
            (books / name / ".draft").write_text("x", encoding="utf-8")
    monkeypatch.setattr(b, "BOOKS", books)
    monkeypatch.setattr(b, "DOCS", tmp_path / "docs")
    assert [slug for slug, _, _ in b.copy_books()] == ["open"]
