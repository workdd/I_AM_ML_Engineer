# OpenAI Machine Learning Engineer 출제 경향

> 외부 문제은행에서 관찰한 출제 유형 정리. 문제 본문과 해설은 옮기지 않고 **제목·난이도·분류만** 기록한다.

- **출처**: [PracHub - OpenAI Machine Learning Engineer](https://prachub.com/companies/openai/positions/machine-learning-engineer)
- **관찰 시점**: 2026-09-08
- **표본**: 목록 첫 페이지 20건 (전체 78건 중)
- **인용 범위**: 사이트 이용약관이 자동 수집을 금지하므로 목록 한 페이지에서 확인한 메타데이터만 기록한다. 본문은 링크로 연결한다.

## 왜 이 파일을 두는가

문제를 모아두려는 목적이 아니다. **어떤 유형이 반복되는지 알면 무엇을 준비할지 정해진다.** 문제 본문은 필요할 때 링크로 열어 보고, 실제 연습은 `experiments/` 노트북에서 직접 구현하는 방식으로 한다.

## 유형 분포

| 분류 | 건수 |
| --- | --- |
| Machine Learning | 10 |
| ML Coding | 7 |
| Coding & Algorithms | 5 |
| Simulation & State Machines | 3 |
| Software Engineering Fundamentals | 3 |
| Annotation & Label Quality | 3 |
| Neural Networks & Training | 2 |
| Transformers & Attention | 2 |
| ML System Design | 2 |
| Model Evaluation | 1 |
| LLM Inference & Serving | 1 |
| LLM Evaluation | 1 |
| AI Agents | 1 |
| Unsupervised Learning | 1 |
| Concurrency | 1 |
| Debugging | 1 |

난이도는 Medium 11건, Hard 9건이다. **Easy가 한 건도 없다.**

## 읽어낸 경향

**1. 순수 알고리즘 문제가 아니다.** `Coding & Algorithms` 5건 중에도 행렬 곱 누적, 격자 시뮬레이션처럼 수치 연산이 섞인 문제가 많다. LeetCode식 자료구조 문제와는 결이 다르다.

**2. ML 개념을 코드로 옮기는 능력을 본다.** 1-NN을 신경망 forward pass로 표현하기, masked cross-entropy에 label smoothing 넣기, 작은 네트워크의 backprop 구현하기가 대표적이다. **개념을 아는 것과 벡터화해서 짜는 것은 다른 능력**이고, 후자를 묻는다.

**3. NumPy 벡터화가 반복해서 나온다.** 파이썬 루프 없이 처리하라는 제약이 자주 붙는다. 메모리가 모자랄 때 어떻게 나눌지까지 묻는 경우가 있다.

**4. 수치 안정성을 따로 본다.** 스트리밍 엔트로피 계산에서의 안정성처럼, 수식대로 짜면 터지는 지점을 아는지 확인한다.

**5. 최근 주제가 섞여 들어온다.** LLM 추론과 서빙, LLM 평가, AI 에이전트, 어노테이션 품질이 각각 잡힌다. 전통적인 ML 문제만 준비하면 빈다.

**6. 라운드가 Technical Screen 중심이다.** 표본 대부분이 초기 기술 스크리닝 단계로 표시돼 있다.

## 문제 목록

| 난이도 | 제목 | 분류 | 푼 사람 |
| --- | --- | --- | --- |
| Medium | [Vectorize One-Nearest-Neighbor and Express It as a Neural Forward Pass](https://prachub.com/interview-questions/vectorize-one-nearest-neighbor-and-express-it-as-a-neural-forward-pass) | Machine Learning, ML Coding, Neural Networks & Training | 614 |
| Medium | [Simulate Infection Spread with Immunity and Recovery](https://prachub.com/coding-questions/simulate-infection-spread-with-immunity-and-recovery) | Coding & Algorithms, Simulation & State Machines | 379 |
| Hard | [Compute Prefix Matrix Products](https://prachub.com/interview-questions/compute-prefix-matrix-products) | Coding & Algorithms | 120 |
| Hard | [Design an In-Memory Key-Value Cache](https://prachub.com/interview-questions/design-an-in-memory-key-value-cache) | Software Engineering Fundamentals | 70 |
| Medium | [Locked](https://prachub.com/interview-questions/implement-masked-cross-entropy-with-label-smoothing) | Machine Learning, ML Coding | 883 |
| Medium | [Compute entropy and implement 1-NN](https://prachub.com/interview-questions/compute-entropy-and-implement-1-nn) | Machine Learning, ML Coding | 1483 |
| Hard | [Improve Training With Noisy Annotators](https://prachub.com/interview-questions/improve-training-with-noisy-annotators) | Machine Learning, Annotation & Label Quality | 1011 |
| Hard | [Locked](https://prachub.com/coding-questions/streaming-entropy-with-numerical-stability) | Streaming Entropy with Numerical Stability, Coding & Algorithms | 95 |
| Medium | [Derive Sharded Matrix Multiplication and Backpropagation](https://prachub.com/interview-questions/derive-sharded-matrix-multiplication-and-backpropagation) | Machine Learning, Neural Networks & Training | 500 |
| Medium | [Locked](https://prachub.com/interview-questions/implement-1nn-with-numpy) | Machine Learning, ML Coding | 2921 |
| Hard | [Compute Matrix Prefix Products And Gradients](https://prachub.com/interview-questions/compute-matrix-prefix-products-and-gradients) | Machine Learning, ML Coding | 1001 |
| Hard | [Improve classifier with noisy multi-annotator labels](https://prachub.com/interview-questions/improve-classifier-with-noisy-multi-annotator-labels) | Machine Learning, Annotation & Label Quality, Model Evaluation | 6391 |
| Hard | [Simulate Plant Infection With Controlled Burning](https://prachub.com/coding-questions/simulate-plant-infection-with-controlled-burning) | Coding & Algorithms, Simulation & State Machines, Grids & Matrices | 105 |
| Medium | [Debug MiniGPT and Backpropagate Matmul](https://prachub.com/interview-questions/debug-minigpt-and-backpropagate-matmul) | Machine Learning, ML Coding, Transformers & Attention | 1191 |
| Hard | [Locked](https://prachub.com/coding-questions/simulate-infection-spread-on-a-grid) | Simulate Infection Spread on a Grid, Coding & Algorithms, Simulation & State Machines | 115 |
| Medium | [Explain KV cache in Transformer inference](https://prachub.com/interview-questions/explain-kv-cache-in-transformer-inference) | Software Engineering Fundamentals, Transformers & Attention, LLM Inference & Serving | 1730 |
| Hard | [Locked](https://prachub.com/interview-questions/implement-backprop-for-a-tiny-network) | Implement Backprop for a Tiny Network, Machine Learning, ML Coding | 1637 |
| Medium | [Mine Novel Images from Unlabeled Data](https://prachub.com/interview-questions/mine-novel-images-from-unlabeled-data) | ML System Design, Unsupervised Learning, Annotation & Label Quality | 322 |
| Medium | [Design an Agent Harness and Evaluation System](https://prachub.com/interview-questions/design-an-agent-harness-and-evaluation-system) | ML System Design, LLM Evaluation, AI Agents | 187 |
| Medium | [Debug a Concurrent Job Scheduler](https://prachub.com/interview-questions/debug-a-concurrent-job-scheduler) | Software Engineering Fundamentals, Concurrency, Debugging | 509 |
## 이 저장소에서의 연습 방법

`interviews/`의 다른 파일은 질문에 답을 쓰는 형식이지만, 이 유형은 **직접 구현해봐야 늘어난다.** `experiments/` 노트북 구조(Pre-Quiz → TODO 구현 → 테스트 → 시각화 → Post-Quiz)를 그대로 쓴다.

우선순위는 분포를 따른다.

- [ ] NumPy 벡터화: 거리 행렬, 브로드캐스팅, 메모리 분할 처리
- [ ] 손실 함수 직접 구현: cross-entropy, label smoothing, masking
- [ ] backprop 수동 구현 (이미 `03_backpropagation.ipynb`에 기반이 있다)
- [ ] 수치 안정성: log-sum-exp, 스트리밍 통계
- [ ] 격자 시뮬레이션과 상태 전이
- [ ] LLM 추론 구조: KV cache, 배칭, 서빙 지표

마지막 항목은 [서빙 구조 노트](../readings/blogs/%5B20260908%5D%20Notion_%EC%84%9C%EB%B9%99%EA%B5%AC%EC%A1%B0%EA%B0%80_LLM_%EC%B2%98%EB%A6%AC%EB%9F%89%EC%9D%84_%EB%B0%94%EA%BE%B8%EB%8A%94_%EB%B0%A9%EC%8B%9D.md)와 이어진다. KV Cache 용량 계산에서 GQA를 빠뜨리면 6배 틀린다는 내용이 거기 있다.
