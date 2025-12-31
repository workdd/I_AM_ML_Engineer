# I AM ML Engineer

> ML/LLM 핵심 개념을 **"왜?"** 에 답할 수 있는 깊이로 정리하는 학습 저장소

[![GitHub issues](https://img.shields.io/github/issues/workdd/I_AM_ML_Engineer)](https://github.com/workdd/I_AM_ML_Engineer/issues)
[![GitHub stars](https://img.shields.io/github/stars/workdd/I_AM_ML_Engineer)](https://github.com/workdd/I_AM_ML_Engineer/stargazers)

## About

ML 관련 내용을 **수학적 원리 이해와 구현으로** 깊이 있게 정리하고 것이 이 프로젝트의 목표입니다.

- 핵심 개념의 직관적 이해 + 수학적 유도
- NumPy/PyTorch Scratch 구현
- 예상 Q&A 포함
- LLM을 충분히 활용하여, 질의 응답 및 내용 정리 진행

## Learning Roadmap

### LLM Core
| Topic | Status | Document |
|-------|--------|----------|
| Transformer & Self-Attention | ![](https://img.shields.io/badge/status-done-brightgreen) | [Notebook](experiments/transformer/) |
| Fine-tuning (LoRA, QLoRA) | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| RAG Architecture | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Tokenizer (BPE, SentencePiece) | ![](https://img.shields.io/badge/status-todo-lightgrey) | - |
| Decoding Strategies | ![](https://img.shields.io/badge/status-done-brightgreen) | [08_mini_gpt](experiments/transformer/08_mini_gpt.ipynb) |

### ML Fundamentals
| Topic | Status | Document |
|-------|--------|----------|
| Gradient Descent & Optimizers | ![](https://img.shields.io/badge/status-done-brightgreen) | [01_gradient_descent](experiments/basics/01_gradient_descent.ipynb) |
| Backpropagation | ![](https://img.shields.io/badge/status-done-brightgreen) | [03_backpropagation](experiments/basics/03_backpropagation.ipynb) |
| Activation Functions | ![](https://img.shields.io/badge/status-done-brightgreen) | [02_activation_functions](experiments/basics/02_activation_functions.ipynb) |
| Regularization (L1/L2/Dropout) | ![](https://img.shields.io/badge/status-done-brightgreen) | [04_regularization](experiments/basics/04_regularization.ipynb) |
| Batch/Layer Normalization | ![](https://img.shields.io/badge/status-done-brightgreen) | [05_batch_norm](experiments/basics/05_batch_norm.ipynb) |
| PCA | ![](https://img.shields.io/badge/status-done-brightgreen) | [06_pca](experiments/basics/06_pca.ipynb) |

## Repository Structure

```
I_AM_ML_Engineer/
├── experiments/           # 실습 노트북 (직접 구현)
│   ├── basics/            # ML/DL 기초 (Optimizer, Backprop 등)
│   └── transformer/       # Transformer 구현 (Attention → GPT)
├── deep_learning/         # Transformer, CNN, RNN 등
├── classical_ml/          # 전통 ML 알고리즘
├── fundamentals/          # 수학적 기초
├── llm_trends/            # LLM 최신 트렌드
├── practical_tips/        # 실무 경험 및 팁
└── readings/              # 읽은 글 정리
```

## Hands-on Experiments

**직접 구현하며 배우는** 학습 노트북입니다. 각 노트북은 Pre-Quiz → TODO 구현 → 테스트 → 시각화 → Post-Quiz → 정답 구조로 되어 있습니다.

### ML/DL Basics (`experiments/basics/`)

| # | Topic | Key Concepts |
|---|-------|--------------|
| 01 | [Gradient Descent](experiments/basics/01_gradient_descent.ipynb) | Vanilla SGD, Momentum, RMSprop, Adam |
| 02 | [Activation Functions](experiments/basics/02_activation_functions.ipynb) | Sigmoid, ReLU, GELU, XOR 문제 |
| 03 | [Backpropagation](experiments/basics/03_backpropagation.ipynb) | Chain Rule, Computational Graph, MLP |
| 04 | [Regularization](experiments/basics/04_regularization.ipynb) | L1/L2, Dropout, Early Stopping |
| 05 | [Batch Normalization](experiments/basics/05_batch_norm.ipynb) | BatchNorm vs LayerNorm |
| 06 | [PCA](experiments/basics/06_pca.ipynb) | 공분산, 고유값 분해, 차원 축소 |

### Transformer Implementation (`experiments/transformer/`)

| # | Topic | Key Concepts |
|---|-------|--------------|
| 01 | [Self-Attention](experiments/transformer/01_self_attention.ipynb) | Scaled Dot-Product, √d_k 스케일링 |
| 02 | [Multi-Head Attention](experiments/transformer/02_multihead_attention.ipynb) | Head 분리/병합, Concat |
| 03 | [Feed Forward](experiments/transformer/03_feed_forward.ipynb) | Position-wise FFN, GELU |
| 04 | [Layer Normalization](experiments/transformer/04_layer_norm.ipynb) | Pre-LN vs Post-LN |
| 05 | [Positional Encoding](experiments/transformer/05_positional_encoding.ipynb) | Sinusoidal PE, 위치 정보 |
| 06 | [Encoder Block](experiments/transformer/06_encoder_block.ipynb) | Residual Connection, 전체 조립 |
| 07 | [Decoder Block](experiments/transformer/07_decoder_block.ipynb) | Causal Mask, GPT 스타일 |
| 08 | [Mini GPT](experiments/transformer/08_mini_gpt.ipynb) | 전체 모델, Text Generation |

## Recent Readings

최근 읽고 정리한 기술 글들입니다.

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2025-12-30 | [AI 테스트 에이전트 구축](readings/blogs/[20251230]%20Medium_AI_테스트_에이전트_구축.md) | Medium | `#TDD` `#AIAgent` `#ClaudeCode` `#SubAgent` |
| 2025-12-30 | [Context Engineering 핵심역량](readings/blogs/[20251230]%20SKdevocean_Context_Engineering_핵심역량.md) | SK devocean | `#ContextEngineering` `#LLM` `#ContextWindow` |
| 2025-12-30 | [OMS Claude AI 워크플로우](readings/blogs/[20251230]%20컬리_OMS_Claude_AI_워크플로우.md) | 컬리 기술블로그 | `#ClaudeAI` `#MSA` `#팀생산성` |
| 2025-12-29 | [Visa Intelligent Commerce + AgentCore](readings/blogs/[20251229]%20AWS블로그_Visa_Intelligent_Commerce_AgentCore.md) | AWS ML Blog | `#AgenticAI` `#Bedrock` `#MCP` `#MultiAgent` |
| 2025-12-27 | [AI 진화: 계산기부터 LLM까지](readings/blogs/[20251227]%20네이버클로바_AI진화_계산기부터_LLM까지.md) | 네이버 CLOVA | `#AI역사` `#딥러닝` `#옴니모달` |
| 2025-12-27 | [Logistic Regression 통계 vs ML](readings/blogs/[20251227]%20Velog_Logistic_Regression_통계vs머신러닝.md) | Velog | `#LogisticRegression` `#MLE` `#SGD` |
| 2025-12-25 | [Claude Code 스타일 스킬 시스템](readings/blogs/[20251225]%20AWS_Strands_스킬시스템_Claude_Code_스타일.md) | AWS Samples | `#LLM` `#Agent` `#Skill-System` |
| 2025-12-24 | [Table Agent 테이블 데이터 처리](readings/blogs/[20251224]%20채널톡_Table_Agent_테이블데이터_처리.md) | 채널톡 | `#RAG` `#Text-to-SQL` `#Agent` |
| 2025-12-24 | [ML 모델 벤치마크 필요성](readings/blogs/[20251224]%20채널톡_ML모델_벤치마크_필요성.md) | 채널톡 | `#RAG` `#벤치마크` `#하이브리드검색` |
| 2025-12-23 | [머신러닝 테스트 코드 구현](readings/blogs/[20251223]%20velog_ML_테스트코드_구현.md) | velog | `#MLOps` `#Testing` `#pytest` |
| 2025-12-22 | [Subagents Supervisor 패턴](readings/blogs/[20251222]%20LangChain_Subagents_Supervisor_패턴.md) | LangChain | `#MultiAgent` `#Supervisor` |
| 2025-12-19 | [LLM 서빙 성능최적화](readings/blogs/[20251219]%20네이버클로바_LLM서빙_성능최적화.md) | 네이버 CLOVA | `#LLM` `#KVCache` `#Goodput` |
| 2025-12-19 | [Speculative Decoding 적용기](readings/blogs/[20251219]%20네이버클로바_Speculative_Decoding_적용기.md) | 네이버 CLOVA | `#LLM` `#SpeculativeDecoding` |
| 2025-12-19 | [Tensor Parallelism 심층분석](readings/blogs/[20251219]%20nanovllm_Tensor_Parallelism_심층분석.md) | liyuan24 블로그 | `#LLM` `#TensorParallel` |
| 2025-12-18 | [토스 대규모 데이터 서빙 아키텍처](readings/blogs/[20251218]%20토스_대규모_데이터서빙_아키텍처.md) | 토스 기술블로그 | `#DataEngineering` `#StarRocks` |
| 2025-12-18 | [MCP vs Claude Skills 비교](readings/blogs/[20251218]%20요즘IT_MCP와_Claude_Skills_비교.md) | 요즘IT | `#MCP` `#ClaudeSkills` |
| 2025-12-18 | [LLM 버그 트리아지 자동화](readings/blogs/[20251218]%20채널톡_LLM_버그트리아지_자동화.md) | 채널톡 | `#LLM` `#자동화` |

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
