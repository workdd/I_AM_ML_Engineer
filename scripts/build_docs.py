#!/usr/bin/env python3
"""readings/ 와 interviews/ 를 MkDocs 가 읽는 docs/ 로 조립한다.

저장소의 파일명 규칙(`[YYYYMMDD] 출처_제목.md`)은 그대로 두고,
빌드할 때만 docs/ 아래로 복사한 뒤 목차(SUMMARY.md)를 생성한다.
docs/ 는 생성물이므로 커밋하지 않는다.
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# (원본 디렉터리, docs 안 경로)
TREES = [
    ("readings", "readings"),
    ("interviews", "interviews"),
    ("practical_tips", "practical_tips"),
]

GITHUB_BLOB = "https://github.com/workdd/I_AM_ML_Engineer/blob/main"
# 사이트에 복사하지 않는 디렉터리. 이쪽 링크는 GitHub 로 돌린다
EXTERNAL_DIRS = ("experiments/", "deep_learning/")

# README 의 주제 절 제목과 docs 안 디렉터리를 잇는다
TOPIC_ORDER = [
    ("그래프와 지식 표현", "graph"),
    ("에이전트와 도구 연결", "agent"),
    ("LLM 서빙과 추론 최적화", "serving"),
    ("신뢰도와 환각 탐지", "trust"),
    ("엔지니어링 일반", "eng"),
]

KW = {
    "graph": ["GraphRAG", "LogicRAG", "ROGRAG", "KnowledgeGraph", "Ontology", "OKF",
              "지식표현", "Leiden", "RAPTOR", "DRIFT", "TextAttributedGraph",
              "GraphNeuralNetwork", "LocalSearch", "GlobalSearch", "MultiHopQA"],
    "serving": ["KVCache", "Goodput", "SpeculativeDecoding", "TensorParallel", "Throughput",
                "vLLM", "RayServe", "Triton", "KServe", "CrossEntropy", "MemoryOptimization",
                "CUDA", "토큰효율", "TOON", "StarRocks", "DataEngineering", "ColdStart",
                "Karpenter", "SleepMode", "SparseAttention", "MoE", "LongContext", "FP4",
                "Quantization", "PTQ", "NVFP4"],
    "trust": ["Confidence", "Logits", "LogProbs", "환각", "BlackBox", "FutureContext",
              "벤치마크", "하이브리드검색", "Verifier", "TestTimeScaling", "BestOfN"],
    "agent": ["Agent", "MCP", "ClaudeSkills", "Skill", "MultiAgent", "Supervisor",
              "ToolSelection", "ContextEngineering", "AgentCore", "DeepAgents",
              "Documentation", "PromptBloat", "AIAgent", "ClaudeCode", "SubAgent",
              "PositionPaper", "LangGraph", "Observability", "Bedrock", "SystemPrompt",
              "PromptDesign", "권한설계", "PromptInjection", "SSE", "Redis", "PubSub"],
}


def title_of(path: Path) -> str:
    """파일의 첫 H1 을 제목으로 쓴다. 없으면 파일명."""
    for line in path.read_text(encoding="utf-8").split("\n")[:40]:
        if line.startswith("# "):
            return line[2:].strip()
    return path.stem


def tags_of(path: Path) -> str:
    head = "\n".join(path.read_text(encoding="utf-8").split("\n")[:30])
    m = re.search(r"^- \*\*태그\*\*:\s*(.+)$", head, re.M)
    return m.group(1) if m else ""


def date_of(path: Path) -> str:
    m = re.match(r"\[(\d{8})\]", path.name)
    return m.group(1) if m else "00000000"


def bucket_of(path: Path) -> str:
    blob = f"{path.name} {title_of(path)} {tags_of(path)}"
    for key in ("graph", "serving", "trust", "agent"):
        if any(w.lower() in blob.lower() for w in KW[key]):
            return key
    return "eng"


MERMAID_FENCE = re.compile(r"(```mermaid\n)(.*?)(```)", re.S)


def _escape_mermaid_br(text: str) -> str:
    """mermaid 펜스 안의 <br/> 를 HTML 엔티티로 바꾼다.

    mermaid2 는 펜스 내용을 escape 없이 내보낸다. 그러면 브라우저가 <br/> 를
    DOM 요소로 파싱해 버리고, mermaid 가 읽는 textContent 에서 사라져 라벨이
    한 줄로 붙는다. 엔티티로 넣으면 textContent 단계에서 <br/> 로 되돌아온다.
    저장소 원본은 건드리지 않으므로 GitHub 자체 렌더도 그대로 동작한다.
    """

    def fix(m: re.Match[str]) -> str:
        body = m.group(2).replace("<br/>", "&lt;br/&gt;").replace("<br>", "&lt;br&gt;")
        return m.group(1) + body + m.group(3)

    return MERMAID_FENCE.sub(fix, text)


def _postprocess(root: Path) -> None:
    for md in root.rglob("*.md"):
        text = md.read_text(encoding="utf-8")
        fixed = _escape_mermaid_br(text)
        if fixed != text:
            md.write_text(fixed, encoding="utf-8")


def copy_trees() -> None:
    if DOCS.exists():
        shutil.rmtree(DOCS)
    DOCS.mkdir(parents=True)
    for src_name, dst_name in TREES:
        src = ROOT / src_name
        if src.exists():
            # README.md 는 목차 역할이라 SUMMARY.md 가 대신한다
            shutil.copytree(src, DOCS / dst_name,
                            ignore=shutil.ignore_patterns("README.md"))
    _postprocess(DOCS)
    index = DOCS / "index.md"
    shutil.copy(ROOT / "README.md", index)
    _link_out(index)


# README 최상단의 중앙 정렬 블록은 사이트 링크와 미리보기라
# 사이트 안에서는 자기 자신을 가리킨다. 간단한 머리말로 갈아 끼운다
HERO = re.compile(r"\A<div align=\"center\">.*?</div>\n+", re.S)
HERO_REPLACEMENT = """# I AM ML Engineer

읽은 것을 **"왜 이렇게 만들었나"** 까지 파고들어 정리하는 저장소

"""


def _link_out(path: Path) -> None:
    """사이트에 없는 경로를 가리키는 링크를 GitHub 원본으로 돌린다."""
    text = HERO.sub(HERO_REPLACEMENT, path.read_text(encoding="utf-8"))

    def repl(m: re.Match[str]) -> str:
        target = m.group(2)
        if target.startswith(EXTERNAL_DIRS):
            return f"[{m.group(1)}]({GITHUB_BLOB}/{target})"
        return m.group(0)

    path.write_text(re.sub(r"\[([^\]]+)\]\(([^)]+)\)", repl, text), encoding="utf-8")


def collect_notes() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {k: [] for _, k in TOPIC_ORDER}
    for p in sorted((DOCS / "readings").rglob("*.md")):
        if p.parent.name == "_maps":
            continue
        groups[bucket_of(p)].append(p)
    for key in groups:
        groups[key].sort(key=date_of, reverse=True)
    return groups


def write_summary(groups: dict[str, list[Path]]) -> None:
    """literate-nav 는 빈 줄 없는 하나의 목록을 요구한다."""
    lines = ["* [홈](index.md)"]

    maps_dir = DOCS / "readings" / "_maps"
    maps = sorted(maps_dir.glob("*.md")) if maps_dir.exists() else []
    if maps:
        lines.append("* 지도")
        for m in maps:
            lines.append(f"    * [{title_of(m)}]({m.relative_to(DOCS).as_posix()})")

    for label, key in TOPIC_ORDER:
        notes = groups.get(key, [])
        if not notes:
            continue
        lines.append(f"* {label}")
        for n in notes:
            lines.append(f"    * [{title_of(n)}]({n.relative_to(DOCS).as_posix()})")

    iv = DOCS / "interviews"
    if iv.exists():
        lines.append("* 면접 준비")
        for f in sorted(iv.glob("*.md")):
            if f.name == "README.md":
                continue
            lines.append(f"    * [{title_of(f)}]({f.relative_to(DOCS).as_posix()})")

    (DOCS / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    copy_trees()
    groups = collect_notes()
    write_summary(groups)
    total = sum(len(v) for v in groups.values())
    print(f"docs/ 조립 완료: 노트 {total}건")
    for label, key in TOPIC_ORDER:
        print(f"  {label}: {len(groups.get(key, []))}건")


if __name__ == "__main__":
    main()
