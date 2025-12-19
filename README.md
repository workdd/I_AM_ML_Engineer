# I AM ML Engineer

> ML/LLM 핵심 개념을 **"왜?"** 에 답할 수 있는 깊이로 정리하는 학습 저장소

[![GitHub issues](https://img.shields.io/github/issues/workdd/I_AM_ML_Engineer)](https://github.com/workdd/I_AM_ML_Engineer/issues)
[![GitHub stars](https://img.shields.io/github/stars/workdd/I_AM_ML_Engineer)](https://github.com/workdd/I_AM_ML_Engineer/stargazers)

## About

ML 관련 내용을 **수학적 원리 이해와 구현으로** 깊이 있게 정리하고 것이 이 프로젝트의 목표입니다.

- 핵심 개념의 직관적 이해 + 수학적 유도
- NumPy/PyTorch Scratch 구현
- 예상 Q&A 포함

## Learning Roadmap

### LLM Core
| Topic | Status | Document |
|-------|--------|----------|
| Transformer & Self-Attention | ![](https://img.shields.io/badge/status-in%20progress-yellow) | [Link]() |
| Fine-tuning (LoRA, QLoRA) | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| RAG Architecture | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Tokenizer (BPE, SentencePiece) | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Decoding Strategies | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |

### ML Fundamentals
| Topic | Status | Document |
|-------|--------|----------|
| Gradient Descent & Optimizers | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Backpropagation | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Bias-Variance Tradeoff | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Regularization (L1/L2) | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |

## Repository Structure

```
I_AM_ML_Engineer/
├── deep_learning/         # Transformer, CNN, RNN 등
├── classical_ml/          # 전통 ML 알고리즘
├── fundamentals/          # 수학적 기초
├── llm_trends/            # LLM 최신 트렌드
├── practical_tips/        # 실무 경험 및 팁
└── readings/              # 읽은 글 정리
```

## Recent Readings

최근 읽고 정리한 기술 글들입니다.

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2024-12-19 | [Tensor Parallelism 심층분석](readings/blogs/[20241219]%20nanovllm_Tensor_Parallelism_심층분석.md) | liyuan24 블로그 | `#LLM` `#TensorParallel` |
| 2024-12-18 | [토스 대규모 데이터 서빙 아키텍처](readings/blogs/[20241218]%20토스_대규모_데이터서빙_아키텍처.md) | 토스 기술블로그 | `#DataEngineering` `#StarRocks` |
| 2024-12-18 | [MCP vs Claude Skills 비교](readings/blogs/[20241218]%20요즘IT_MCP와_Claude_Skills_비교.md) | 요즘IT | `#MCP` `#ClaudeSkills` |
| 2024-12-18 | [LLM 버그 트리아지 자동화](readings/blogs/[20241218]%20채널톡_LLM_버그트리아지_자동화.md) | 채널톡 | `#LLM` `#자동화` |

## Document Template

각 주제는 다음 구조로 정리됩니다:

1. **한 줄 정의** - 간단히 설명 가능한 정의
2. **직관적 이해** - 수식 없이 개념 설명
3. **수학적 원리** - 핵심 수식 유도
4. **Scratch 구현** - NumPy/PyTorch 기반 구현
5. **Q&A** - 예상 질문과 답변

## Contributing

학습 내용에 대한 피드백이나 토론은 언제든 환영합니다!
[Issues](https://github.com/workdd/I_AM_ML_Engineer/issues)에 의견을 남겨주세요.

## License

MIT License
