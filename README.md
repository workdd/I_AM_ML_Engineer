# I AM ML Engineer

> 읽은 것을 **"왜 이렇게 만들었나"** 까지 파고들어 정리하는 저장소

[![GitHub issues](https://img.shields.io/github/issues/workdd/I_AM_ML_Engineer)](https://github.com/workdd/I_AM_ML_Engineer/issues)
[![GitHub stars](https://img.shields.io/github/stars/workdd/I_AM_ML_Engineer)](https://github.com/workdd/I_AM_ML_Engineer/stargazers)

## About

논문, 기술 블로그, 오픈소스 저장소를 읽고 정리합니다. 요약이 아니라 **판단에 쓸 수 있는 형태**로 남기는 것이 목표입니다.

정리마다 다음을 지킵니다.

- **원문 확인**: 수치와 인용은 원문에서 직접 확인합니다. 2차 자료를 옮겨 적지 않습니다
- **한계 명시**: 어떤 조건에서 나온 결과인지, 어디까지 일반화되는지 함께 적습니다
- **상호 연결**: 같은 문제를 다루는 자료끼리 링크로 잇습니다
- **적용 지점**: 읽은 내용 중 실제로 쓸 수 있는 부분을 따로 뽑습니다

## Readings

### 그래프와 지식 표현

GraphRAG 계열과 지식 그래프. 서로 반박하고 보완하는 관계라 묶어서 읽는 편이 낫습니다.

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2026-09-10 | [Graph RAG의 모든 것: 패턴과 구현체 지형도](readings/blogs/[20260910]%20devto_Graph_RAG의_모든_것.md) | dev.to | `#GraphRAG` `#RAPTOR` `#DRIFT` `#KnowledgeGraph` `#AWS` |
| 2026-09-10 | [LLM on Graphs 서베이: GraphRAG는 이 지형의 한 칸이다](readings/papers/[20260910]%20LLM_on_Graphs_Comprehensive_Survey.md) | arXiv | `#Survey` `#GraphNeuralNetwork` `#GraphRAG` `#TextAttributedGraph` |
| 2026-09-08 | [Microsoft GraphRAG 동작원리 단계별 해부](readings/blogs/[20260908]%20TowardsAI_Microsoft_GraphRAG_동작원리_단계별.md) | Towards AI | `#GraphRAG` `#KnowledgeGraph` `#Leiden` `#LocalSearch` `#GlobalSearch` |
| 2026-08-31 | [ROGRAG: 다단계 검색 GraphRAG](readings/papers/[20260831]%20ROGRAG_Robustly_Optimized_GraphRAG.md) | arXiv / GitHub | `#GraphRAG` `#LogicForm` `#KnowledgeGraph` `#Ablation` |
| 2026-07-02 | [LogicRAG: Adaptive Reasoning Structures](readings/papers/[20260702]%20LogicRAG_Adaptive_Reasoning_Structures.md) | arXiv / GitHub | `#RAG` `#GraphRAG` `#MultiHopQA` `#DAG` |
| 2026-06-14 | [Open Knowledge Format(OKF)](readings/blogs/[20260614]%20PyTorchKR_Open_Knowledge_Format_OKF.md) | PyTorchKR | `#OKF` `#지식표현` `#LLMWiki` `#GoogleCloud` `#MCP` |

### 에이전트와 도구 연결

MCP, 멀티 에이전트, 도구 선택, 에이전트 운영 인프라.

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2026-09-10 | [에이전트 중단 기능 설계: Flag 체크포인트](readings/blogs/[20260910]%20Liner_에이전트_중단기능_설계.md) | Liner | `#AIAgent` `#SSE` `#Redis` `#PubSub` `#상태관리` |
| 2026-09-02 | [Bedrock AgentCore: LangGraph 위에 운영 레이어 얹기](readings/blogs/[20260902]%20AWS블로그_Bedrock_AgentCore_멀티데이터소스_NLP_에이전트.md) | AWS 기술 블로그 | `#AgentCore` `#Bedrock` `#LangGraph` `#MCP` `#Observability` |
| 2026-09-02 | [The End of Software Engineering: 에이전트 패러다임 선언 검토](readings/papers/[20260902]%20End_of_Software_Engineering_Agentic_Paradigm.md) | arXiv | `#AgenticEngineering` `#AIAgent` `#PositionPaper` `#SWEbench` |
| 2026-07-10 | [RAG-MCP: Prompt Bloat 없는 MCP 도구 선택](readings/papers/[20260710]%20RAG-MCP_Prompt_Bloat_Tool_Selection.md) | arXiv | `#MCP` `#RAG` `#ToolSelection` `#PromptBloat` |
| 2026-07-06 | [Agentic AI: Single vs Multi-Agent Systems](readings/blogs/[20260706]%20Medium_Agentic_AI_Single_vs_Multi_Agent_Systems.md) | Medium | `#AgenticAI` `#MultiAgent` `#LangGraph` `#WorkflowDesign` |
| 2026-07-06 | [How Many Tools Should an LLM Agent See?](readings/papers/[20260706]%20How_Many_Tools_Should_an_LLM_Agent_See.md) | arXiv | `#LLMAgent` `#ToolSelection` `#Retrieval` `#MCP` `#BoR` |
| 2026-07-02 | [OpenWiki: repo 문서화 에이전트](readings/blogs/[20260702]%20LangChain_OpenWiki_Repo_Documentation_Agent.md) | LangChain Blog | `#OpenWiki` `#AIAgent` `#Documentation` `#DeepAgents` |
| 2025-12-31 | [AI 테스트 에이전트 구축](readings/blogs/[20251231]%20Medium_AI_테스트_에이전트_구축.md) | Medium | `#TDD` `#AIAgent` `#ClaudeCode` `#SubAgent` |
| 2025-12-30 | [Context Engineering 핵심역량](readings/blogs/[20251230]%20SKdevocean_Context_Engineering_핵심역량.md) | SK devocean | `#ContextEngineering` `#LLM` `#ContextWindow` |
| 2025-12-30 | [OMS Claude AI 워크플로우](readings/blogs/[20251230]%20컬리_OMS_Claude_AI_워크플로우.md) | 컬리 기술블로그 | `#ClaudeAI` `#MSA` `#팀생산성` |
| 2025-12-29 | [Visa Intelligent Commerce + AgentCore](readings/blogs/[20251229]%20AWS블로그_Visa_Intelligent_Commerce_AgentCore.md) | AWS ML Blog | `#AgenticAI` `#Bedrock` `#MCP` `#MultiAgent` |
| 2025-12-25 | [Claude Code 스타일 스킬 시스템](readings/blogs/[20251225]%20AWS_Strands_스킬시스템_Claude_Code_스타일.md) | AWS Samples | `#LLM` `#Agent` `#Skill-System` |
| 2025-12-24 | [Table Agent 테이블 데이터 처리](readings/blogs/[20251224]%20채널톡_Table_Agent_테이블데이터_처리.md) | 채널톡 | `#RAG` `#Text-to-SQL` `#Agent` |
| 2025-12-22 | [Subagents Supervisor 패턴](readings/blogs/[20251222]%20LangChain_Subagents_Supervisor_패턴.md) | LangChain | `#MultiAgent` `#Supervisor` |
| 2025-12-18 | [MCP vs Claude Skills 비교](readings/blogs/[20251218]%20요즘IT_MCP와_Claude_Skills_비교.md) | 요즘IT | `#MCP` `#ClaudeSkills` |

### LLM 서빙과 추론 최적화

처리량, 지연, 메모리. 엔진 설정과 서빙 구조를 나눠 보는 관점.

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2026-09-11 | [DeepSeek-V4.1-Flash: KV 캐시를 토큰당 890바이트로](readings/papers/[20260911]%20DeepSeek-V4.1-Flash_KV_Cache_Compression.md) | DeepSeek-AI | `#KVCache` `#SparseAttention` `#MoE` `#LongContext` `#FP4` |
| 2026-09-10 | [EKS에서 vLLM 콜드 스타트 428초를 226초로](readings/blogs/[20260910]%20AWS블로그_EKS_vLLM_Gemma4_콜드스타트_최적화.md) | AWS 기술 블로그 | `#vLLM` `#EKS` `#ColdStart` `#Karpenter` `#SleepMode` |
| 2026-09-08 | [서빙 구조가 LLM 처리량을 바꾸는 방식](readings/blogs/[20260908]%20Notion_서빙구조가_LLM_처리량을_바꾸는_방식.md) | Notion | `#vLLM` `#RayServe` `#Triton` `#KServe` `#Throughput` |
| 2026-07-06 | [Fused Linear Cross-Entropy: 메모리 아끼면서 CE 계산하기](readings/blogs/[20260706]%20TrillionLabs_Fused_Linear_Cross_Entropy.md) | Trillion Labs Research | `#CrossEntropy` `#LLMTraining` `#MemoryOptimization` `#CUDA` |
| 2026-01-04 | [JSON vs TOON 토큰효율](readings/blogs/[20260104]%20네이버클라우드_JSON_vs_TOON_토큰효율.md) | 네이버클라우드 | `#TOON` `#JSON` `#토큰효율` `#LLM` |
| 2025-12-19 | [LLM 서빙 성능최적화](readings/blogs/[20251219]%20네이버클로바_LLM서빙_성능최적화.md) | 네이버 CLOVA | `#LLM` `#KVCache` `#Goodput` |
| 2025-12-19 | [Speculative Decoding 적용기](readings/blogs/[20251219]%20네이버클로바_Speculative_Decoding_적용기.md) | 네이버 CLOVA | `#LLM` `#SpeculativeDecoding` |
| 2025-12-19 | [Tensor Parallelism 심층분석](readings/blogs/[20251219]%20nanovllm_Tensor_Parallelism_심층분석.md) | liyuan24 블로그 | `#LLM` `#TensorParallel` |
| 2025-12-18 | [토스 대규모 데이터 서빙 아키텍처](readings/blogs/[20251218]%20토스_대규모_데이터서빙_아키텍처.md) | 토스 기술블로그 | `#DataEngineering` `#StarRocks` |

### 신뢰도와 환각 탐지

답을 믿어도 되는지 판단하는 신호들.

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2026-06-11 | [Future Context 기반 LLM 환각 검출](readings/blogs/[20260611]%20네이버클로바_Future_Context_환각검출.md) | 네이버 CLOVA | `#환각검출` `#BlackBox` `#FutureContext` `#ACL2026` |
| 2026-06-09 | [Logits as Confidence: LLM·VLM 신뢰도 활용](readings/blogs/[20260609]%20Medium_Logits_as_Confidence_LLM_VLM.md) | Medium | `#Logits` `#Confidence` `#LLM` `#VLM` `#LogProbs` |
| 2025-12-24 | [ML 모델 벤치마크 필요성](readings/blogs/[20251224]%20채널톡_ML모델_벤치마크_필요성.md) | 채널톡 | `#RAG` `#벤치마크` `#하이브리드검색` |

### 엔지니어링 일반

| 날짜 | 제목 | 출처 | 태그 |
|------|------|------|------|
| 2025-12-27 | [AI 진화: 계산기부터 LLM까지](readings/blogs/[20251227]%20네이버클로바_AI진화_계산기부터_LLM까지.md) | 네이버 CLOVA | `#AI역사` `#딥러닝` `#옴니모달` |
| 2025-12-27 | [Logistic Regression 통계 vs ML](readings/blogs/[20251227]%20Velog_Logistic_Regression_통계vs머신러닝.md) | Velog | `#LogisticRegression` `#MLE` `#SGD` |
| 2025-12-23 | [머신러닝 테스트 코드 구현](readings/blogs/[20251223]%20velog_ML_테스트코드_구현.md) | velog | `#MLOps` `#Testing` `#pytest` |
| 2025-12-18 | [LLM 버그 트리아지 자동화](readings/blogs/[20251218]%20채널톡_LLM_버그트리아지_자동화.md) | 채널톡 | `#LLM` `#자동화` |

## Repos & Tools

직접 뜯어본 오픈소스 저장소와 도구입니다. 코드 규모, 기여자, 라이선스, 실제 구현 여부까지 확인해 적습니다.

| 날짜 | 레포 | 조직 | 태그 |
|------|------|------|------|
| 2026-09-11 | [GPT-6 Astra 프롬프트 설계 분석](readings/repos/[20260911]%20CL4R1T4S_GPT-6_Astra_프롬프트_설계_분석.md) | CL4R1T4S | `#SystemPrompt` `#PromptDesign` `#권한설계` `#PromptInjection` |
| 2026-09-11 | [llm-as-a-verifier: logprob 분포로 에이전트 궤적 채점](readings/repos/[20260911]%20llm-as-a-verifier_학습없이_에이전트_궤적을_채점하는_검증_프레임워크.md) | llm-as-a-verifier | `#Verifier` `#LogProbs` `#TestTimeScaling` `#BestOfN` `#PrefixCache` |
| 2026-09-02 | [reef — 서빙하면서 스스로 학습하는 에이전트 인프라](readings/repos/[20260902]%20Human-Agent-Society_reef_자기개선_에이전트_지속학습_인프라.md) | Human-Agent-Society | `#ContinualLearning` `#SelfImprovingAgent` `#RL` `#HarnessEvolution` `#SGLang` |
| 2026-07-31 | [MatrAIx-Persona-8B — 페르소나 인구규모 AI 평가 인프라](readings/repos/[20260731]%20MatrAIx-ai_MatrAIx-Persona-8B_페르소나_인구규모_AI평가인프라.md) | MatrAIx-ai | `#PersonaSimulation` `#AI평가` `#SyntheticData` `#LLMAgent` `#Benchmark` |
| 2026-07-23 | [AgentENV — 에이전트 환경 대규모 실행 플랫폼](readings/repos/[20260723]%20kvcache-ai_AgentENV_에이전트환경_대규모실행.md) | kvcache-ai | `#AgenticRL` `#Firecracker` `#Sandbox` `#E2B` `#KimiK3` |
| 2026-04-11 | [late.sh — SSH 한 줄로 들어가는 터미널 클럽하우스](readings/repos/[20260411]%20mpiorowski_late-sh_SSH로_들어가는_터미널_클럽하우스.md) | mpiorowski | `#SSH` `#TUI` `#Rust` `#LLMFirstDocs` `#Kubernetes` |
| 2026-03-22 | [OpenCrab — 에이전트에 MCP로 붙이는 온톨로지 공장](readings/repos/[20260322]%20AlexAI-MCP_OpenCrab_MCP로_붙이는_온톨로지_공장.md) | AlexAI-MCP | `#MCP` `#Ontology` `#KnowledgeGraph` `#GraphRAG` `#ReBAC` |

## Interview Prep

`interviews/` 에 질문별 답변과 기업별 출제 경향을 정리합니다.

| 파일 | 내용 |
|------|------|
| [01_통계_수학.md](interviews/01_통계_수학.md) | 선형대수, 확률, 통계 |
| [02_머신러닝.md](interviews/02_머신러닝.md) | ML 알고리즘과 개념 |
| [03_딥러닝.md](interviews/03_딥러닝.md) | 딥러닝 일반 |
| [04_자연어처리.md](interviews/04_자연어처리.md) | NLP |
| [14_OpenAI_MLE_출제경향.md](interviews/14_OpenAI_MLE_출제경향.md) | OpenAI ML Engineer 문제 유형 분포 |

## Hands-on Experiments

직접 구현하며 확인하는 노트북입니다. Pre-Quiz → TODO 구현 → 테스트 → 시각화 → Post-Quiz 구조입니다.

| 분류 | 노트북 |
|------|--------|
| ML/DL 기초 | [Gradient Descent](experiments/basics/01_gradient_descent.ipynb) · [Activation](experiments/basics/02_activation_functions.ipynb) · [Backprop](experiments/basics/03_backpropagation.ipynb) · [Regularization](experiments/basics/04_regularization.ipynb) · [BatchNorm](experiments/basics/05_batch_norm.ipynb) · [PCA](experiments/basics/06_pca.ipynb) |
| Transformer | [Attention → Mini GPT](experiments/transformer/) (8개 노트북, [08_mini_gpt](experiments/transformer/08_mini_gpt.ipynb) 에서 Decoding Strategies 포함) |

## Repository Structure

```
I_AM_ML_Engineer/
├── readings/              # 읽은 자료 정리 (이 저장소의 중심)
│   ├── papers/            # 논문
│   ├── blogs/             # 기술 블로그
│   └── repos/             # 오픈소스 저장소·도구
├── interviews/            # 면접 질문 정리와 기업별 출제 경향
├── experiments/           # 실습 노트북 (직접 구현)
│   ├── basics/            # ML/DL 기초
│   └── transformer/       # Attention 에서 GPT 까지
├── deep_learning/         # Transformer, CNN, RNN 개념 정리
└── practical_tips/        # 실무 경험과 팁
```

## 정리 형식

파일명은 `[YYYYMMDD] 출처_제목.md` 로 통일합니다. 각 정리는 다음 뼈대를 따릅니다.

1. **메타데이터** - 원문 링크, 저자, 발행일, 실험 환경, 읽은 날짜, 태그
2. **한 줄 요약** - 이 자료가 주장하는 것
3. **본문** - 아래 본문 규칙을 따릅니다
4. **읽을 때 감안할 것** - 조건, 표본, 검증되지 않은 전제, 발견한 오기
5. **내 작업과의 연결** - 실제로 가져다 쓸 지점
6. **결론** - 이 자료를 어떻게 쓰는 것이 맞는지

4번이 핵심입니다. 벤더 자료의 정량 지표 부재, 단일 환경 실험, 인용 오류 같은 것을 여기 적습니다.

본문 규칙
--

**논문은 원문 목차를 그대로 따라갑니다.** 절 번호와 제목을 원문과 같게 두어 대조하며 읽을 수 있게 합니다. 임의로 재구성하지 않습니다. 4번과 5번 절만 읽는 쪽이 덧붙인 것으로 표시합니다.

블로그와 저장소 정리는 원문 구조를 따르되 주제 단위로 묶습니다. 원문이 절 번호를 쓰지 않는 경우가 많고, 흐름이 설명에 맞춰져 있기 때문입니다.

공통으로 지키는 것은 다음과 같습니다.

- 수치와 표는 원문에서 직접 확인해 옮깁니다. 2차 자료의 인용을 재인용하지 않습니다
- 원문에서 발견한 오기나 불일치는 4번 절에 기록합니다
- 같은 문제를 다루는 다른 정리와 상대 경로로 링크를 겁니다

## Contributing

정리 내용에 대한 피드백이나 토론은 환영합니다. [Issues](https://github.com/workdd/I_AM_ML_Engineer/issues)에 남겨주세요.

## License

MIT License
