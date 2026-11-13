#!/usr/bin/env bash
# books/<책>/ 원고를 위키독스 GitHub 연동 책 저장소로 보낸다.
#
#   scripts/wikidocs_sync.sh <책 폴더 이름> <owner/repo> [--push]
#
# 원고는 이미 위키독스 형식(README.md · TOC.md · pages/ · assets/, 페이지 H1 없음)이라
# 그대로 복사한다. 원본은 이 저장소이고 위키독스 저장소는 사본이다. 저쪽을 직접 고치지 않는다.
# --push 가 없으면 커밋까지만 하고 차이를 보여 준다.
set -euo pipefail
BOOK="$1"; REPO="$2"; PUSH="${3:-}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$ROOT/books/$BOOK"
[ -f "$SRC/TOC.md" ] || { echo "TOC.md 없음: $SRC" >&2; exit 1; }

# TOC 에 적힌 페이지가 모두 있는지 먼저 본다
missing=0
while IFS= read -r rel; do
  [ -f "$SRC/$rel" ] || { echo "TOC 에 있는데 파일 없음: $rel" >&2; missing=1; }
done < <(sed -n 's/^ *- \[[^]]*\](\(pages\/[^)]*\)).*/\1/p' "$SRC/TOC.md")
[ "$missing" = 0 ] || exit 1

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
gh repo clone "$REPO" "$WORK/repo" -- -q
cd "$WORK/repo"
# 사본 쪽 콘텐츠를 원고로 통째로 바꾼다 (.git · .github · skills 등 다른 파일은 둔다)
rm -rf pages assets README.md TOC.md
cp "$SRC/README.md" "$SRC/TOC.md" .
cp -R "$SRC/pages" pages
[ -d "$SRC/assets" ] && [ -n "$(ls -A "$SRC/assets")" ] && cp -R "$SRC/assets" assets || mkdir -p assets
git add -A
if git diff --cached --quiet; then echo "변경 없음"; exit 0; fi
SHA="$(git -C "$ROOT" rev-parse --short HEAD)"
git commit -q -m "sync: I_AM_ML_Engineer books/$BOOK @ $SHA"
git --no-pager show --stat --oneline HEAD | head -40
if [ "$PUSH" = "--push" ]; then git push -q origin HEAD && echo "push 완료: $REPO"; else echo "(--push 없이 실행: push 하지 않음)"; fi
