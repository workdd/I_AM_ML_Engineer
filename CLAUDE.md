# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ML 이론 공부부터 코드 구현, 그리고 **실무에서 얻은 경험과 팁**까지 정리하는 저장소입니다.

## Repository Structure (Recommended)

```
I_AM_ML_Engineer/
├── fundamentals/          # 기초 수학 및 통계 (선형대수, 확률, 최적화)
├── classical_ml/          # 전통 ML 알고리즘 (regression, SVM, trees, etc.)
├── deep_learning/         # 딥러닝 (CNN, RNN, Transformer, etc.)
├── nlp/                   # 자연어처리
├── cv/                    # 컴퓨터 비전
├── mlops/                 # MLOps 관련 (배포, 모니터링, 파이프라인)
├── papers/                # 논문 구현 및 정리
├── interviews/            # ML 인터뷰 준비
├── experiments/           # 실험 코드
├── practical_tips/        # 실무 경험 및 팁
├── llm_trends/            # LLM 트렌드 및 도구 학습
└── readings/              # 읽은 글 정리 (블로그, LinkedIn 등)
```

## Development Commands

```bash
# Python 가상환경 설정
python -m venv venv
source venv/bin/activate  # macOS/Linux

# 의존성 설치
pip install -r requirements.txt

# Jupyter 노트북 실행
jupyter notebook

# 테스트 실행
pytest tests/ -v

# 단일 테스트 파일 실행
pytest tests/test_specific.py -v

# 특정 테스트 함수 실행
pytest tests/test_file.py::test_function_name -v
```

## Code Style

- Python: PEP 8 준수
- Type hints 사용 권장
- Docstring: Google style 사용
- 노트북 파일명: `01_topic_name.ipynb` 형식으로 번호 prefix 사용

## ML Implementation Guidelines

### 알고리즘 구현 시
1. 수학적 배경/이론 먼저 마크다운으로 정리
2. NumPy로 직접 구현 (scratch implementation)
3. scikit-learn/PyTorch 등 프레임워크 버전과 비교
4. 시각화 및 실험 결과 포함

### 파일 네이밍
- 이론 정리: `theory_*.md` 또는 `*_theory.ipynb`
- 구현 코드: `impl_*.py` 또는 `*_implementation.py`
- 실험/분석: `exp_*.ipynb` 또는 `*_experiment.ipynb`

## Common Libraries

```python
# 기본
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ML
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Deep Learning
import torch
import torch.nn as nn
```

## Practical Tips (실무 팁)

실무에서 겪은 경험과 노하우를 정리합니다.

### 카테고리
- **debugging/**: 디버깅 및 트러블슈팅 경험
- **performance/**: 성능 최적화 (학습 속도, 추론 최적화)
- **data/**: 데이터 전처리, 품질 관리, 라벨링 노하우
- **deployment/**: 배포 및 서빙 관련 경험
- **team/**: 협업, 코드 리뷰, 실험 관리 방법론
- **gotchas/**: 자주 겪는 실수 및 함정 (pitfalls)

### 실무 팁 작성 형식
```markdown
# [제목]

## 상황 (Context)
어떤 상황에서 발생했는지

## 문제 (Problem)
구체적인 문제 또는 이슈

## 해결 (Solution)
어떻게 해결했는지, 코드 예시 포함

## 교훈 (Lesson Learned)
이 경험에서 배운 점
```

### 파일 네이밍
- `tip_YYYYMMDD_간단한설명.md`
- 예: `tip_20241215_gpu_memory_leak.md`

## LLM Trends (LLM 트렌드)

최신 LLM 관련 기술, 도구, 트렌드를 학습하고 정리합니다.

### 카테고리
- **agents/**: LLM 에이전트 (AutoGPT, CrewAI, LangGraph 등)
- **rag/**: RAG 파이프라인 및 벡터 DB
- **automation/**: n8n, Zapier, Make 등 워크플로우 자동화
- **prompting/**: 프롬프트 엔지니어링 기법
- **fine_tuning/**: 파인튜닝 및 PEFT (LoRA, QLoRA 등)
- **serving/**: LLM 서빙 (vLLM, TGI, Ollama 등)
- **evaluation/**: LLM 평가 방법론

### 주요 도구/프레임워크
| 카테고리 | 도구 |
|---------|------|
| 오케스트레이션 | LangChain, LlamaIndex, Haystack |
| 에이전트 | CrewAI, AutoGen, LangGraph |
| 자동화 | n8n, Flowise, Dify |
| 벡터 DB | Pinecone, Weaviate, Chroma, Milvus |
| 모니터링 | LangFuse, LangSmith, Helicone |

### 파일 네이밍
- `llm_YYYYMMDD_주제.md`
- 예: `llm_20241218_n8n_workflow_setup.md`

## Reading Notes (읽은 글 정리)

LinkedIn, 블로그, 논문 등에서 읽은 유용한 글들을 정리합니다.

### 디렉토리 구조
```
readings/
├── blogs/           # 기술 블로그 글
├── linkedin/        # LinkedIn 포스트
├── papers/          # 논문 요약 (papers/와 별도로 간단 요약용)
└── newsletters/     # 뉴스레터 (The Batch, TLDR AI 등)
```

### 작성 형식
```markdown
# [글 제목]

- **출처**: [링크](URL)
- **저자**: 저자명
- **읽은 날짜**: YYYY-MM-DD
- **태그**: #tag1 #tag2

## 핵심 내용
- 요점 1
- 요점 2

## 인상 깊은 부분
> 인용문이나 핵심 문장

## 내 생각 / 적용점
이 글에서 배운 점, 실무에 적용할 수 있는 부분
```

### 파일 네이밍
- `read_YYYYMMDD_출처_제목요약.md`
- 예: `read_20241218_linkedin_llm_production_tips.md`
