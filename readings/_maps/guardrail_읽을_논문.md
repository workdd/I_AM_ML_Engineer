# LLM 가드레일 읽을 논문

"정상 질문의 분포를 잡아 두고, 거기서 벗어난 입력을 위반으로 본다"는 아이디어를 중심으로 모은 읽기 대기열입니다. 정상 분포와의 거리(A)를 먼저 보고, 같은 내부 표현을 지도학습으로 쓰는 쪽(B), 더 얕은 신호(C, D), 비교 기준이 되는 분류기형 가드레일(E) 순으로 묶었습니다.

- **작성**: 2026-10-07
- **수집 방법**: 웹 검색 5회. arXiv ID와 제목, 제출 연월은 arXiv API로 확인했습니다
- **주의**: 요지 칸은 검색 결과의 초록 요약을 옮긴 것이고 본문은 확인하지 않았습니다. 정리할 때 원문으로 다시 확인합니다

## 진행 방법

1. 아래 표에서 `대기` 중 우선순위가 가장 높은 논문을 고릅니다
2. `readings/papers/[YYYYMMDD] 제목.md`로 README의 정리 형식을 따라 정리합니다
3. 이 표의 상태를 `정리 완료`로 바꾸고 제목에 노트 링크를 겁니다
4. 읽어 보니 정리할 가치가 없으면 상태를 `보류`로 두고 이유를 한 줄 적습니다

상태 값은 `대기`, `읽는 중`, `정리 완료`, `보류` 네 가지입니다.

## 추천 순서

```
A1 → A2 → A6 → B2 → 나머지
```

- **A1 kNNGuard**: 아이디어와 가장 가깝습니다. 학습 없이 은닉 상태에서 kNN 거리로 판정하고, 참조 예시만 바꿔 새 도메인에 맞춥니다
- **A2 Revisiting JBShield**: 표현 수준 방어를 공격으로 깨 보고 다시 세웁니다. "분포 밖이면 잡힌다"는 전제가 적응형 공격에서 얼마나 버티는지 봅니다
- **A6 Off-topic 탐지**: 용도가 좁은 챗봇에서 범위 밖 질문을 막는 문제입니다. 정상 분포가 좁을수록 이 접근이 잘 맞는다는 가설을 확인합니다
- **B2 Jailbreaking Leaves a Trace**: 탈옥 입력이 내부 표현에 남기는 흔적을 층별로 분석합니다. 어느 층을 써야 하는지 정하는 근거가 됩니다

## A. 정상 분포와의 거리 (OOD 탐지)

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| A1 | 1 | kNNGuard: Turning LLM Hidden Activations into a Training-Free Configurable Guardrail | [2607.02072](https://arxiv.org/abs/2607.02072) | 안전·위험 예시 50개의 은닉 상태로 참조 묶음을 만들고 여러 층의 kNN 점수로 판정. 재학습 없이 참조 묶음 교체로 도메인 적응. 6개 도메인에서 평균 F1 최고 | 대기 |
| A2 | 1 | Revisiting JBShield: Breaking and Rebuilding Representation-Level Jailbreak Defenses | [2605.03095](https://arxiv.org/abs/2605.03095) | 표현 기반 탈옥 방어를 깨고 다시 설계. 정상 묶음까지의 kNN 거리와 Mahalanobis 거리 판정을 다룸 | 대기 |
| A3 | 2 | Rethinking Jailbreak Detection of LVLMs with Representational Contrastive Scoring | [2512.12069](https://arxiv.org/abs/2512.12069) | 안전에 결정적인 층 하나에서 정상과 악성 분포까지의 상대 거리로 판정 (RCS). 멀티모달 모델 대상 | 대기 |
| A4 | 2 | BERM: Low-Overhead Prompt-Injection Detection via In-Situ Benign Representation Modeling (IJCAI 2026) | [IJCAI](https://www.ijcai.org/proceedings/2026/68) | 응답하는 모델의 내부 표현을 그대로 써서 대조학습 분류기로 판정. 비교 방법 대비 F1 5.2%p 향상, 12배 이상 빠름 | 대기 |
| A5 | 3 | Embedding-based classifiers can detect prompt injection attacks | [2410.22284](https://arxiv.org/abs/2410.22284) | 임베딩 모델 3종으로 정상과 악성의 분포 차이를 보고 지도 분류기 학습. Random Forest + OpenAI 임베딩이 AUC 0.764 | 대기 |
| A6 | 1 | A Flexible LLM Guardrail Development Methodology Applied to Off-Topic Prompt Detection | [2411.12946](https://arxiv.org/abs/2411.12946) | 실제 데이터 없이 LLM으로 범위 밖 질문을 합성해 벤치마크 겸 학습 데이터로 씀 | 대기 |
| A7 | 2 | Guarded Query Routing for Large Language Models | [2505.14524](https://arxiv.org/abs/2505.14524) | 질의를 도메인별로 보내면서 범위 밖·위험 질의를 걸러 내는 라우팅 | 대기 |
| A8 | 3 | Domain Certification (ICLR 2025) | [ICLR](https://proceedings.iclr.cc/paper_files/paper/2025/file/27befed4547edcb4bdeacef9472cadee-Paper-Conference.pdf) | 적대적 공격 아래에서 범위 밖 응답 확률의 상한을 보장하는 VALID 알고리즘 | 대기 |

## B. 내부 표현을 지도학습으로 읽기 (probe, 개념 방향)

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| B1 | 2 | JBShield: Defending LLMs from Jailbreak Attacks through Activated Concept Analysis and Manipulation | [2502.07557](https://arxiv.org/abs/2502.07557) | 유해 개념과 탈옥 개념의 활성화를 분석해 탐지하고 조작. 5개 모델, 9개 공격에서 평균 정확도 0.95, F1 0.94. A2가 이 방법을 공격함 | 대기 |
| B2 | 1 | Jailbreaking Leaves a Trace: Understanding and Detecting Jailbreak Attacks from Internal Representations | [2602.11495](https://arxiv.org/abs/2602.11495) | 은닉 활성화의 구조를 텐서 기반으로 잡아 미세조정이나 별도 LLM 없이 탐지 | 대기 |
| B3 | 3 | Do Internal Layers of LLMs Reveal Patterns for Jailbreak Detection? | [2510.06594](https://arxiv.org/abs/2510.06594) | 탈옥과 정상 입력에 대한 은닉 층 반응 비교. B2와 같은 저자의 앞선 연구 | 대기 |
| B4 | 2 | What Features in Prompts Jailbreak LLMs? Investigating the Mechanisms Behind Attacks | [2411.03343](https://arxiv.org/abs/2411.03343) | 은닉 상태 probe로 탈옥 성공 예측. 분포 안에서는 정확하지만 탈옥마다 내부 메커니즘이 달라 단일 방향이 없음 | 대기 |
| B5 | 2 | Refusal in Language Models Is Mediated by a Single Direction | [2406.11717](https://arxiv.org/abs/2406.11717) | 거절 행동이 잔차 스트림의 한 방향으로 매개됨. probe 설계의 배경 지식 | 대기 |
| B6 | 2 | PIShield: Detecting Prompt Injection Attacks via Intrinsic LLM Features | [2510.14005](https://arxiv.org/abs/2510.14005) | LLM 내부 특징으로 프롬프트 인젝션 탐지 | 대기 |
| B7 | 2 | Attention Tracker: Detecting Prompt Injection Attacks in LLMs (NAACL 2025 Findings) | [2411.00348](https://arxiv.org/abs/2411.00348) | 일부 어텐션 헤드가 원래 지시에서 주입된 지시로 주의를 옮기는 현상을 추적. 학습 없음, AUROC 최대 10.0% 개선 | 대기 |

## C. 토큰 수준 신호 (perplexity)

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| C1 | 2 | Detecting Language Model Attacks with Perplexity | [2308.14132](https://arxiv.org/abs/2308.14132) | 적대적 접미사가 붙은 질의의 약 90%가 perplexity 1000 초과. perplexity와 토큰 길이로 LightGBM 학습해 오탐 해소 | 대기 |
| C2 | 3 | Baseline Defenses for Adversarial Attacks Against Aligned Language Models | [2309.00614](https://arxiv.org/abs/2309.00614) | perplexity 필터, 바꿔 쓰기, 재토큰화 같은 기본 방어 비교 | 대기 |
| C3 | 3 | Token-Level Adversarial Prompt Detection Based on Perplexity Measures and Contextual Information | [2311.11509](https://arxiv.org/abs/2311.11509) | 문장 전체가 아니라 토큰 단위로 perplexity와 주변 문맥을 보고 적대적 구간 표시 | 대기 |

## D. 그래디언트와 손실 지형

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| D1 | 3 | GradSafe: Detecting Jailbreak Prompts for LLMs via Safety-Critical Gradient Analysis (ACL 2024) | [2402.13494](https://arxiv.org/abs/2402.13494) | 탈옥 질의에 순응 응답을 붙였을 때 안전 관련 파라미터의 그래디언트가 비슷한 패턴을 보이는 점을 이용 | 대기 |
| D2 | 3 | Gradient Cuff: Detecting Jailbreak Attacks by Exploring Refusal Loss Landscapes | [2403.00867](https://arxiv.org/abs/2403.00867) | 거절 손실의 값과 매끄러움으로 2단계 탐지 | 대기 |

## E. 비교 기준: 분류기형 가드레일

| ID | 우선 | 논문 | arXiv | 요지 | 상태 |
|---|---|---|---|---|---|
| E1 | 2 | Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations | [2312.06674](https://arxiv.org/abs/2312.06674) | 위험 분류 체계로 미세조정한 LLM이 입력과 출력을 판정. 대부분 논문의 비교 대상 | 대기 |
| E2 | 2 | Constitutional Classifiers: Defending against Universal Jailbreaks across Thousands of Hours of Red Teaming | [2501.18837](https://arxiv.org/abs/2501.18837) | 헌법(규칙 문서)으로 합성한 데이터로 입출력 분류기 학습. 대규모 레드팀 결과 | 대기 |
