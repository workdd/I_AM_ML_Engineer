# ROGRAG: 질의 분해가 실패하면 fuzzy matching으로 내려가는 GraphRAG

- **출처**: [GitHub - tpoisonooo/ROGRAG](https://github.com/tpoisonooo/ROGRAG)
- **논문**: [ROGRAG: A Robustly Optimized GraphRAG Framework](https://arxiv.org/abs/2503.06474)
- **저자**: Zhefan Wang, Huanjun Kong, Jie Ying, Wanli Ouyang, Nanqing Dong
- **소속**: Shanghai Artificial Intelligence Laboratory, Shanghai Innovation Institute, The Chinese University of Hong Kong
- **발표**: ACL 2025 demo track (저장소 설명 기준)
- **arXiv**: 2025-03-09 v1 제출, 2025-06-04 v2
- **라이선스**: BSD-3-Clause
- **읽은 날짜**: 2026-08-31
- **태그**: #GraphRAG #RAG #KnowledgeGraph #LogicForm #Ablation #Retrieval

## 한 줄 요약

질의를 연산자로 분해해 그래프를 밟는 logic form 검색을 먼저 시도하고 분해나 검증이 실패하면 dual-level fuzzy matching으로 격하(degrade)해 답을 만드는 다단계 GraphRAG. 새 알고리즘보다는 기존 GraphRAG 구현 네 개를 하나로 합친 뒤 ablation으로 부품별 기여도를 재는 데 무게가 실려 있다.

## 이름이 두 개인 이유

이 저장소는 이력이 조금 헷갈린다. arXiv 2503.06474는 v1에서 `HuixiangDou2: A Robustly Optimized GraphRAG Approach`였고 v2(2025-06)에서 제목이 `ROGRAG: A Robustly Optimized GraphRAG Framework`로 바뀌면서 저자 목록도 달라졌다. 저장소 이름도 HuixiangDou2에서 ROGRAG로 옮겨왔고 파이썬 패키지명은 지금도 `huixiangdou`다. 검색하다 세 이름이 섞여 나와도 같은 프로젝트다.

## 문제의식 - GraphRAG는 성능보다 평가가 먼저 막힌다

논문이 지적하는 세 가지는 성능 개선이 아니라 검증 가능성 문제다.

- **부품 기여도를 분리할 수 없다.** 엔티티 추출, 그래프 구축, 질의 분해, 검색, 생성이 서로 물려 있어 어느 단계가 점수를 올렸는지 말하기 어렵다.
- **공개 벤치마크가 이미 학습 데이터에 섞여 있다.** 논문은 UltraDomain 데이터셋으로 이걸 직접 확인했다. Qwen2.5-7B-Instruct의 직답과 정답(GT)을 Kimi API로 비교시켰더니 도메인별 LLM win rate가 0.65~0.97로 나왔다. 7B 모델의 무보강 답변이 정답보다 선호됐다. 응답 길이는 GT의 6.6~10.8배였다.
- **질의 분해가 heuristic이라 잘못되면 그대로 무너진다.** LLM이 subquery를 틀리게 만들면 이후 검색 전체가 오염된다.

두 번째 항목이 이 논문의 실험 설계를 결정한다. LLM이 이미 잘 푸는 데이터셋에서 RAG 점수가 올라가는 건 검색이 좋아졌다는 증거가 못 되므로 vanilla 점수가 낮은 도메인 데이터셋을 일부러 골랐다. 그래서 평가는 종자·농업 도메인 벤치마크인 SeedBench에서 돌린다.

## 핵심 아이디어 - 한 번의 검색으로 끝내지 않는다

ROGRAG의 검색은 단계가 있고 아래로 내려갈수록 정밀도 대신 견고함을 택한다.

```text
사용자 질의
  -> LLM이 intent / domain 판별
  -> [1단] logic form 검색: 연산자 시퀀스로 분해해 그래프 실행
  -> verifier: 이 context로 질문에 답할 수 있는가?
       통과 -> 생성
       실패 -> [2단] dual-level 검색: low/high-level 키워드 fuzzy matching
  -> 검증된 context로 LLM 답변 생성
```

분해가 성공하면 근거가 또렷한 답을 얻고 실패하면 흐릿하지만 뭐라도 잡아오는 쪽으로 떨어진다. 논문 표현으로는 "질의 분해를 우선하고, 분해나 검증이 실패하면 fuzzy matching으로 격하한다". 이 구조 덕에 스트리밍 응답이 끊기지 않는다는 점도 저자들이 강조한다.

베이스가 되는 구현 넷을 각각 어디서 가져왔는지도 논문에 명시돼 있다. DB-GPT는 확장성, LightRAG는 단순한 구현, KAG는 추론, HuixiangDou는 견고함(거절 응답, intent slot)이다.

## 방법론 흐름

### 1. 인덱싱: preprocess -> NER -> dump

코퍼스를 정규화·청킹한 뒤 청크마다 엔티티, 관계, 키워드, 설명, weight를 뽑아 그래프 노드/엣지를 만들고 원본 청크로 역링크를 건다. 논문 예시는 `(scientist, discovery; Marie Curie discovered radium in 1898; 4.5)` 형태다.

청킹 설정은 부록에 있다. 중국어는 ChineseRecursiveTextSplitter, 영어는 RecursiveCharacterTextSplitter에 overlap 32, 기본 chunk size는 768 토큰이다. 저장은 TuGraph, 엔티티·관계 임베딩은 BCEmbedding, 색인은 Faiss를 쓴다.

### 2. Loop NER: 빠뜨린 엔티티를 다시 캐묻는 방식

GraphRAG 계열은 재현율을 올리려고 LLM에게 "더 뽑을 게 있냐"를 반복해서 묻는다. 논문은 두 구현을 비교한다. trial 버전은 NER 후 더 필요한지 물어보고 긍정이면 진행하는 평범한 순서고 base 버전은 "엔티티가 더 남아 있을 것"이라고 먼저 알려준 뒤 추가 추출을 시킨다.

결과는 노드·엣지 수가 많을수록 정확도가 올라가는 쪽이었다. 오답 엔티티는 고립 노드로 남아 그래프 구조와 최종 정확도에 거의 영향을 주지 않는다는 게 저자들의 해석이다. 재현율을 공격적으로 밀어도 된다는 뜻이라 실무에서 쓸모 있는 관찰이다.

| 버전 | 노드 | 엣지 | 정확도 |
|------|------|------|--------|
| Trial | 20,739 | 19,857 | 0.61 |
| Base | 21,838 | 26,847 | 0.69 |
| Optimize Prompt | 29,086 | 35,750 | **0.74** |

### 3. Dual-level 검색: 낮은 층과 높은 층을 따로 뽑는다

질의를 두 갈래로 분해한다. low-level 키워드는 엔티티, high-level 키워드는 관계를 가리키는 서술이다. 엔티티는 fuzzy matching으로 노드에 붙여 연결된 엣지를 끌어오고 관계 키워드는 엣지에 붙여 연결된 노드를 끌어온다. 두 결과를 합친 뒤 중복 노드·엣지·청크를 걷어내 최종 context를 만든다. LightRAG의 방식을 이어받은 부분이다.

### 4. Logic form 검색: 질의를 연산자 시퀀스로 바꾼다

KAG 계열의 추론 검색이다. 미리 정의한 연산자 집합으로 질의를 subquery 리스트로 분해하고 subquery마다 대응 연산자를 골라 실행한 뒤 (subquery, sub-answer) 쌍을 history로 누적한다.

저장소 코드(`service/retriever/logic/`)에서 실제로 파싱하는 연산자는 `get`, `get_spo`/`retrieval`, `count`, `sum`, `sort`, `compare`이고 실행기는 그래프 실행기와 수식 실행기로 나뉜다. 논문이 예로 든 "Zhefu 802가 부모보다 몇 cm 큰가" 같은 계산 질문은 dual-level로는 답이 안 나오고 이 경로라야 풀린다.

### 5. Verifier: 답을 만들기 전에 볼 것인가, 만든 뒤에 볼 것인가

검증 시점을 두 가지로 나눠 비교한다. **argument checking**은 답변 생성 전에 "지금 context로 이 질문에 답할 수 있는가"만 본다. **result checking**은 질문·context·답변을 함께 놓고 전체 정합성을 본다.

결과는 argument checking이 낫다. 저자들의 해석은 두 갈래다. 추론 측면에서는 이미 생성된 답변이 LLM의 주의를 나눠 가져가 질문 자체에 집중하지 못하게 만든다. 모델 측면에서는 Qwen2.5-7B-Instruct 같은 causal 모델은 context 안에서 추론이 맞으면 결과도 대체로 맞아 사후 검증이 중복이다.

## 실험 결과

주 실험은 SeedBench 네 개 서브셋에서 Qwen2.5-7B-Instruct로 돌렸다. BM25와 RQ-RAG는 FlashRAG 구현을 썼다.

| Method | QA-1 (Accuracy) | QA-2 (F1) | QA-3 (Rouge) | QA-4 (Rouge) |
|--------|-----------------|-----------|--------------|--------------|
| vanilla (w/o RAG) | 0.57 | 0.71 | 0.16 | 0.35 |
| LangChain | 0.68 | 0.68 | 0.15 | 0.04 |
| BM25 | 0.65 | 0.69 | 0.23 | 0.03 |
| RQ-RAG | 0.59 | 0.62 | 0.17 | 0.33 |
| **ROGRAG** | **0.75** | **0.79** | **0.36** | **0.38** |

눈에 띄는 건 QA-2다. RAG를 붙인 세 baseline이 모두 무보강 LLM보다 낮다. 관련 없는 context가 오히려 모델을 흔든 결과로, LLM이 익숙하지 않은 도메인에서는 검색을 붙인다고 항상 좋아지지 않는다는 사례다. LangChain과 BM25가 생성 과제인 QA-4에서 0.04, 0.03으로 무너진 건 파라미터 설정 탓에 관련 내용이 거의 안 걸린 결과로 저자들은 본다.

이후 ablation은 SeedBench QA-1 객관식만 써서 정확도로 잰다.

| 실험 | 설정 | 정확도 |
|------|------|--------|
| 최대 context 길이 | 32k | 0.67 |
| | 64k | 0.65 |
| 매칭 방식 | Dual-level | 0.650 |
| | + 28k context length | 0.690 |
| | + expand low-level keys | 0.695 |
| | Exact matching | 0.635 |
| 검색 방식 | Optimized dual-level (평균 9,863자) | 0.74 |
| | Logic form (평균 1,699자) | 0.55 |
| 검증 방식 | Argument checking | **0.75** |
| | Result checking | 0.72 |

읽을 때 놓치기 쉬운 세 가지가 있다.

- **context는 길수록 좋지 않다.** YaRN 스케일을 키워 32k에서 64k로 늘리자 정확도가 2%p 떨어졌다. 정보량이 늘어 엔티티·관계를 뽑아내기 어려워졌다는 해석이다. 청크 크기도 줄이는 쪽이 나았다.
- **exact matching은 dual-level보다 나빴다.** 엔티티를 개별 저장해 정확히 맞추는 방식이 0.635로 오히려 떨어졌다. 질의에 명시되지 않은 암묵적 키워드를 놓치기 때문이다. fuzzy matching이 이 계열 검색의 본질이라는 게 저자들의 결론이다.
- **logic form은 정확도로 채택된 게 아니다.** 0.55로 dual-level의 0.74보다 한참 낮다. 그런데도 1단에 놓은 이유는 출력이 6분의 1 길이로 짧고 논리 전개가 또렷해 도메인 전문가가 그쪽 답변을 선호했기 때문이다. 정확도와 납득 가능성을 갈라서 본 판단이다.

초록의 "60.0%에서 75.0%로"라는 수치는 Table 2의 vanilla 0.57이 아니라 Figure 1의 초기 통합 시스템(a) 점수에서 출발한 값이다. 두 숫자를 같은 baseline으로 섞어 인용하지 않는 게 좋다.

## 저장소 구현 메모

- 패키지 진입점은 `huixiangdou/`이고 v1의 Wechat/Lark/Web 프론트엔드가 그대로 살아 있다. `pipeline/parallel.py`의 `generate()` 시그니처도 v1과 호환되게 유지했다.
- 검색기는 `service/retriever/` 아래 dense, bm25, inverted, knowledge, logic, web으로 나뉘어 있어 단계별로 갈아 끼우기 좋은 구조다.
- Docker(CMD / Swagger API / Gradio)와 소스 설치를 모두 지원하고 CPU 전용과 GPU 설정 예시가 따로 있다. 2025년 11월에 multi-database 지원과 gradio UI 리팩터가 들어갔다.
- HuixiangDou 대비 약 18k 라인 차이라고 README가 밝힌다. 사실상 재작성에 가깝다.
- **테스트 데이터는 라이선스 문제로 공개하지 않는다.** 코드와 결론만 제공한다고 README에 명시돼 있어 수치 재현은 불가능하다.
- SeedLLM이라는 농업 연구 플랫폼에 실제 배포돼 있고 GraphGen 프로젝트에도 그래프 구축 부분이 흘러들어갔다.

## 장점

1. **부품 단위로 재볼 수 있게 만들었다.** GraphRAG 논문 대부분이 파이프라인 전체 점수만 보고하는 데 비해 NER 루프·context 길이·매칭 granularity·검증 시점을 하나씩 갈아 끼우며 잰 기록이 남아 있다. 수치보다 이 기록이 이 논문의 실제 값어치다.
2. **벤치마크 오염을 먼저 측정하고 시작한다.** UltraDomain에서 7B 직답이 GT를 이긴다는 걸 보이고 도메인 데이터셋으로 옮겨간 순서가 정직하다.
3. **격하 경로가 설계에 들어 있다.** 질의 분해 실패를 예외가 아니라 정상 흐름으로 다룬다. agentic RAG를 운영에 올릴 때 실제로 필요한 성질이다.
4. **API 호환을 지키며 갈아엎었다.** 내부를 18k 라인 바꾸면서 외부 인터페이스를 유지한 건 이미 붙어 있는 채널을 끊지 않겠다는 선택이다.

## 한계와 리스크

- **검증 범위가 좁다.** 모델은 Qwen2.5-7B-Instruct 하나, 데이터셋은 SeedBench 한 도메인뿐이다. 저자들도 Limitations에서 인정한다. 다른 모델·도메인에서 같은 ablation 결론이 유지된다는 보장은 없다.
- **평가 벤치마크가 같은 그룹에서 나왔다.** SeedBench는 Shanghai AI Lab 계열 open-sciencelab의 산출물이고 저자 중 Jie Ying이 SeedBench 논문에도 들어간다. 오염 없는 도메인을 고른 판단은 타당하지만 자체 벤치마크 단일 평가라는 점은 감안해야 한다.
- **verifier가 병목이라고 저자들이 직접 말한다.** 정확도 높은 검증기를 만들기 어렵다고 결론과 Limitations에 모두 적혀 있다. argument checking 0.75 대 result checking 0.72는 3%p 차이라 결정적 해법으로 보기도 어렵다.
- **오류 전파는 그대로다.** 엔티티 추출, 질의 분해, 검색 어디서든 틀리면 뒤로 번진다. 다단계 격하는 실패를 덜 치명적으로 만들 뿐 없애지 못한다.
- **재현이 막혀 있다.** 테스트 데이터 비공개라 표의 숫자를 직접 확인할 방법이 없다. 인용할 때 "저자 보고 기준"을 붙이는 편이 안전하다.
- **logic form의 낮은 정확도를 그대로 안고 간다.** 1단이 0.55라는 건 상당수 질의가 2단으로 떨어진다는 뜻이다. 그만큼 verifier 판정이 자주 호출되고 판정이 헐거우면 잘못된 1단 결과가 그대로 통과한다.

## 내 작업과의 연결

1. **격하 경로를 명시적으로 설계하기**

   agentic RAG를 만들 때 질의 분해 실패는 반드시 생긴다. 이때 빈손으로 끝내는 대신 어디까지 내려가서 무엇을 반환할지 미리 정해두는 편이 낫다. ROGRAG는 그 경로를 logic form -> verifier -> dual-level 한 줄로 고정해뒀다.

2. **검증은 답변 전에 붙이는 게 싸고 낫다**

   argument checking이 result checking보다 나았다는 결과는 우리 파이프라인에도 그대로 옮길 만하다. 답변을 만든 뒤 다시 판정하면 토큰도 두 배로 쓰고 판정 품질도 떨어진다. 여기에 logprob confidence 같은 별도 신호를 얹으면 LLM 판정 하나에만 기대는 구조에서 벗어날 수 있다.

3. **NER 재현율은 공격적으로 가져가도 된다**

   오답 엔티티가 고립 노드로 남아 정확도를 크게 해치지 않는다는 관찰은 그래프 구축 단계의 프롬프트 튜닝 방향을 정해준다. 정밀도를 지키려 추출을 보수적으로 하는 것보다 많이 뽑고 나중에 병합하는 쪽이 유리하다.

4. **LogicRAG와 비교해서 볼 것**

   [LogicRAG 노트](%5B20260702%5D%20LogicRAG_Adaptive_Reasoning_Structures.md)와 정반대 지점에 서 있다. LogicRAG는 사전 그래프를 아예 만들지 말고 질의 시점에 구조를 세우자는 쪽이고 ROGRAG는 그래프를 제대로 만들되 검색 단계를 여러 겹으로 쌓자는 쪽이다. 지식베이스가 자주 바뀌면 전자가, 도메인이 고정돼 있고 그래프를 한 번 잘 만들어두면 되는 환경이면 후자가 유리하다.

## 결론

ROGRAG는 새 알고리즘을 제안하는 논문이 아니다. GraphRAG 구현 넷을 하나로 합친 뒤 "어느 부품이 실제로 점수를 올리는가"를 표로 남긴 엔지니어링 리포트에 가깝다. 그래서 SeedBench 0.75라는 숫자보다 ablation 표들이 더 오래 쓸모 있다. context는 길수록 나쁘고, exact matching은 fuzzy보다 못하며, 검증은 답변 전에 붙이는 게 낫다는 세 결론은 다른 GraphRAG를 튜닝할 때도 먼저 확인해볼 만한 가설이다.

다만 단일 모델·단일 도메인 검증에 테스트 데이터까지 비공개라 수치 자체의 일반화는 조심해야 한다. 코드를 참고 구현으로 읽고 결론은 우리 데이터에서 다시 재보는 방식이 맞다.
