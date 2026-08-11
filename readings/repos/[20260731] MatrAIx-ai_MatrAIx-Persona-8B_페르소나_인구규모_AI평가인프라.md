# MatrAIx-Persona-8B — 시뮬레이션된 인구 집단으로 AI 제품을 평가한다

- **레포**: [MatrAIx-ai/MatrAIx-Persona-8B](https://github.com/MatrAIx-ai/MatrAIx-Persona-8B) · [공식 사이트](https://matraix.ai/) · [핸드북](https://github.com/MatrAIx-ai/MatrAIx-Persona-8B/tree/main/docs)
- **제작**: MatrAIx-ai
- **공개일**: 2026-07-31 (Python, MIT, ⭐557 · fork 73 · 2026-08-11 기준)
- **읽은 날짜**: 2026-08-11
- **태그**: #PersonaSimulation #AI평가 #SyntheticData #LLMAgent #ComputerUse #Benchmark

## 이름 함정부터

8B 파라미터 페르소나 모델을 기대하고 들어가면 헛짚는다. 레포 이름의 `8B`는 파라미터 수가 아니라 84억(8.4B) 행짜리 페르소나 코퍼스 `Persona8B`를 가리킨다. 실제로 이 레포는 모델 가중치가 아니라 **데이터셋 + 평가 실행 인프라**를 내놓는다.

## 핵심 문제의식

AI 제품을 평가할 때 우리는 보통 "평균적 사용자" 하나를 상정한다. 프롬프트 하나, 테스트 케이스 하나, 잘해야 사람 몇 명의 정성 피드백이다. 그런데 실제 사용자는 나이·지역·직무·기술 숙련도·성향이 전부 다르고, 제품이 깨지는 지점은 대개 평균에서 먼 쪽에서 먼저 드러난다.

MatrAIx는 이 간극을 **이질적인 시뮬레이션 사용자 집단**으로 메운다. 샘플링한 페르소나 레코드를 LLM 에이전트로 인스턴스화해 재현 가능한 태스크에 그대로 태워 돌린다. 개별 응답과 궤적을 서브그룹·인구 단위 결과로 다시 묶는 리포팅까지가 한 세트다.

README가 스스로 선을 긋는 대목이 오히려 신뢰가 간다. 이름은 영화 *매트릭스*에서 따왔고, 탐색·스트레스 테스트·가설 생성에 쓸 시뮬레이션 세계일 뿐 실제 사람에게서 나온 증거를 대체하지 않는다고 명시한다.

## 설계 축 4가지

| 축 | 방법 | 수치 |
|----|------|------|
| 사람을 구조화 | 1,290개 범주형 차원 스키마 (배경·심리·역량·행동) | 43개 카테고리, 공식 택소노미 9그룹 / 35서브카테고리 |
| 규모 확보 | Full-DAG 그래프 기반 합성 + 실제 데이터 근거 추출 | 100억 생성 → 84억 코퍼스, 공개 코어셋 100만 |
| 태스크 다양성 | Survey · Chat · Web · OS App 네 환경 | macOS·iOS·Linux 네이티브 앱까지 포함 |
| 결과 연결 | 공유 텔레메트리 + 태스크 소유 검증기 | 개별 궤적 → 서브그룹 → 인구 단위 집계 |

## 1,290차원 스키마와 Treiver 추출기

스키마(`persona/schema/dimensions.json`)는 모든 단계가 읽는 계약이다. 연령대·지역·성 정체성·도시화 같은 핵심 인구통계부터 143개 기술 숙련도, 커뮤니케이션·리더십·분석력, 관심사·가치관, 성격 특질까지 각 차원이 정해진 범주값을 갖는다. 페르소나 하나는 차원마다 값 하나를 갖되, 추출로 채워지는 건 일부뿐이다.

자유 서술을 스키마 속성으로 바꾸는 게 **Treiver**(trait-retriever)다. 2단계 경량 RAG다.

```
prompt ──▶ [1] 정규식 검색 ──▶ 후보 차원 ──▶ [2] LLM judge ──▶ 속성
                │                                    │
                └────────── 정규식 속성 ─────────────┘
```

- **1단계 정규식** — 차원별 허용값과 별칭으로 패턴을 만들어 매칭한다. 완전 오프라인·결정적이라 API 키가 필요 없다. "expert"만으로는 무엇에 능한지 모르니, 143개 이상의 숙련도 차원에는 주제어("Data science")까지 함께 걸리도록 **토픽 게이팅**을 건다. 정밀도는 높고 재현율은 낮은 후보를 뽑는다
- **2단계 LLM judge** — Claude가 후보 차원과 허용값만 보고 값 하나를 고르거나 `null`을 반환한다. 원문에서 근거를 인용하고 신뢰도를 함께 매긴다

두 단계 모두 `(dimension_id, value, evidence, method, confidence)` 형태로 떨어져 추출 품질 리포팅에 바로 들어간다. 1단계만 떼어 결정적 속성 추출기로 써도 된다.

## 데이터 구성 — 60% 사람 근거, 40% 합성

공개 코어셋 100만 행의 출처별 내역이다.

| 소스 | 행 수 | 비고 |
|------|-------|------|
| Wikipedia 인물 페이지 | 323,438 | 190만 추출분에서 캘리브레이션 샘플링 |
| Amazon Reviews 2023 | 97,915 | 리뷰 이력 추출, 잔존분 전량 |
| Stack Overflow 서베이 | 113,120 | 응답자 프로필 전량 |
| GSS / World Values Survey | 63,532 | 서베이 응답 매핑 |
| PRISM Alignment | 1,487 | 인간 인터뷰 정렬본 |
| Real Human Survey | 508 | 직접 수집 |
| Full-DAG 합성 | 400,000 | 84억 합성 풀에서 캘리브레이션 샘플링 |

"human-grounded"는 실제 프로필·이력·서베이 레코드에서 왔다는 뜻이지, 추출된 속성 하나하나가 검증된 사실이라는 뜻은 아니다. 뜻을 이렇게 좁혀 정의해둔 점이 눈에 띈다. 모델 추출은 틀릴 수 있고 서베이 매핑은 크로스워크 품질에 달려 있다고 스스로 적어놨다.

인간 데이터 추출에는 `Qwen/Qwen3.6-35B-A3B`(멀티모달 하이브리드 어텐션 MoE, 텍스트 전용으로 사용)를 vLLM 0.24.0 이상으로 돌린다. 원본은 게이티드 HF SQLite(약 7.9GB, 213만 프로필)이다. SLURM 배열 잡으로 샤딩해 재개 가능하게 굴린다.

## 후처리 체인 — 100억에서 100만까지

```
스키마 (1,290차원)
  │
  ├─ curation ─────────── 속성 풀 구축 + 외부 소스 정제 (Wiki·Amazon·서베이)
  │
  ├─ human_extraction ─── 실제 프로필 → 1,290차원 페르소나 (vLLM)
  │
  ├─ synthesis ────────── Full-DAG 그래프 → 100억 합성 페르소나 (CPU 전용 SLURM)
  │
  └─ post_process
        quality_filter ─── 모순 규칙 → 리젝트 비트맵
        deduplication ──── 인간 MinHash + 합성 projection dedup
        unified_dataset ── 잔존 행 Parquet 물질화 (Persona8B)
        coreset_1m ─────── 캘리브레이션된 100만 공개 코어셋
        dataset_statistics 논문용 집계 프로파일링
```

각 단계는 4TB짜리 원본 코드를 다시 쓰는 대신 샤드별 리젝트 비트맵만 뱉는다. 체인 전체가 **비파괴적**이다. 저장 비용이 어느 정도인지도 가늠이 된다. 페르소나당 압축 404바이트 정도라 100억이면 약 4TB, codes만으로도 6~8TB를 잡으라고 한다.

중복 제거를 인간 데이터와 합성 데이터에 다르게 적용한 게 이 파이프라인에서 가장 공학적인 판단이다.

- **인간 산출물**(약 229만 행) — 정확 일치는 128비트 정준 해시로 병합하고 MinHash LSH(순열 64개, 8밴드 × 8행)로 후보를 만든 뒤 시그니처 일치 임계(기본 0.95, 즉 61개 이상 동일)를 넘겨야 병합한다. 시그니처가 임계와 무관하게 재사용되므로 임계를 바꿔도 병합만 다시 돌리면 된다
- **합성 산출물** — MinHash를 쓰지 않는다. 1,290개 코드가 빽빽이 찬 벡터에는 가중 해밍 유사도 `S(x,y)=Σ wᵢ·1[xᵢ=yᵢ]/Σ wᵢ`가 맞는다. 다만 이걸 매번 계산하는 대신 **좌표 투영**을 쓴다. 그래프 사전확률 엔트로피 순으로 16개 이하 필드를 골라 `uint64` 하나로 정확히 인코딩하는 방식이다. 투영 폭은 HyperLogLog 스케치(precision 20, 상대오차 약 0.102%)로 카디널리티를 미리 재서 고른다. 문서가 "투영 일치는 다양성 버킷이지 전체 벡터 유사도 주장이 아니다"라고 못 박는다

최종 총량은 `8,400,000,000 − 인간_dedup_잔존`으로 역산해 결정적 64비트 우선순위 컷오프로 정확히 맞춘다. 컷오프는 65,536빈 히스토그램으로 경계 빈을 찾아 전체 정렬 없이 구한다.

## 캘리브레이션 — 재현 가능한 표본 추출

100만 코어셋은 관측된 주변분포에 맞춘 **비복원 제약 캘리브레이션**으로 뽑는다.

1. 모순 필터와 0.95 MinHash dedup 적용
2. 작은 인간 소스 5종(PRISM·GSS·Survey·Stack Overflow·Amazon)은 잔존분 전량 포함, Wikipedia만 전역 목표에 맞춰 캘리브레이션
3. 전역 목표를 **합성 잔차**로 변환해 인간 데이터가 못 채운 커버리지를 합성으로 보완한다. 차원 `d`의 값 `v`에 대해 `r_dv = p_dv·(H_d + 400,000) − h_dv`, 음수 잔차는 클리핑하고 클리핑된 질량은 infeasibility로 **보고한다(숨기지 않는다)**
4. 후보마다 공유 가중치 하나를 연령 → 지역 → 성 정체성 → 도시화 순으로 곱셈 갱신(`w_i ← w_i·T_dv/E_dv`). 한 번 쓸고 지나가면 앞 마진이 흔들리므로 최대 200회 반복. 기대 개수는 고정 크기 포함확률 `π_i = 1 − e^(−t·w_i)`로 계산하고 `Σ π_i = n`이 되도록 `t`를 푼다
5. 가중치를 정확한 크기의 표본으로 바꿀 때는 **결정적 지수 경주**를 쓴다. 우선순위 `q_i = −log(U_i)/w_i`에서 `U_i`를 시드와 안정적 행 ID로 유도하고 작은 것부터 `n`개를 취한다. 입력 순서에 무관하고 재현 가능하다(시드 `20260720`)

캘리브레이션 근거는 UN World Population Prospects 2024와 World Bank 데이터다. 다만 성 정체성은 UN이 여성·남성만 앵커하므로 나머지 소수 꼬리는 스키마 사전확률(중간 신뢰도)이고, 도시화도 중간 신뢰도, **언어**는 아예 하드 캘리브레이션하지 않는다고 밝힌다. 결측은 절대 대치하지 않는다. `audit.json`과 `RESULTS.md`에 차원별 목표 대비 달성 비율과 절대 오차가 다 남는다.

## 실행 구조

```
Playground UI (localhost:5173)  /  harbor CLI
      │
      ▼
  generate_application_job.py ──► job recipe YAML (에이전트 + 모델 고정)
      │
      ▼
  Matraix Playground 런타임 (uv run harbor run -c ...)
      │
      ├─ 페르소나 샘플링 ── persona-1m Parquet + postings.sqlite 인덱스
      │
      ├─ 에이전트 ── persona-json-survey / persona-user-sim (호스트)
      │              persona-openhands-sdk / browser-use / cocoa / computer-1 (Docker)
      │
      └─ 검증기 ── 태스크가 소유, 궤적·응답 채점 → jobs/<job_name>/
```

용어 정리가 핸드북에 잘 되어 있다. **Task**는 시나리오, **Trial**은 페르소나 하나 × 태스크 하나, **Job**은 YAML 하나에서 나오는 다수 trial 묶음이다. Agent(사용자를 굴리는 하네스)와 Model(페르소나를 연기하는 LLM)이 분리되어 있는 것도 깔끔하다.

## 사용법

```bash
# 요구사항: Docker, uv + Python 3.12, Node.js 20+ (Playground 프런트)
git clone https://github.com/MatrAIx-ai/MatrAIx-Persona-8B.git && cd MatrAIx-Persona-8B
uv venv --python 3.12 && uv pip install -e .
uv pip install -e packages/playground -e packages/harbor-langsmith -e packages/rewardkit

# 스모크 테스트 — API 키 불필요 (Docker 이미지 빌드에 몇 분)
uv run harbor run -c configs/jobs/example-job-recipe/harbor-smoke-local.yaml

# 페르소나 100만 코어셋 임포트 (선택)
huggingface-cli download MatrAIx2026/MatrAIx_Persona_1M_Public_Release \
  --repo-type dataset --local-dir persona/datasets/matraix-persona-1m/release

# 페르소나 1명 × 서베이 태스크 실행
export ANTHROPIC_API_KEY="sk-ant-..."
uv run python application/scripts/generate_application_job.py \
  --task application/tasks/example-survey_product-feedback \
  --execution-mode auto --persona-ids 0042 \
  --model-name anthropic/claude-sonnet-4-6
uv run harbor run -c configs/jobs/application-task-job-recipe/<생성된>-auto-n1.yaml
```

로컬 실습용 페르소나 풀은 `persona/datasets/matraix-persona-dev-sample/`(200명, 스모크용 `0042`)이 따로 들어 있어서 100만 행을 안 받아도 손을 댈 수 있다. Playground를 띄우면 데이터셋 선택 → 코호트 샘플링(최대 1만 행) → 태스크 선택 → 파이프라인 잠금 → 실행 순서로 GUI에서 같은 잡을 굴린다. 태스크를 새로 만들 때는 `application/tasks/example-*`를 복사해 `task.toml` / `instruction.md` / `input/` / 검증기를 고치면 된다.

## 제약

- **실제 사람의 증거를 대체하지 않는다** — README가 직접 밝힌다. 탐색·스트레스 테스트·가설 생성용이다
- **human-grounded ≠ 사실 검증됨** — 모델 추출 오류와 서베이 크로스워크 품질 문제가 그대로 남는다
- **2026-07-20 Persona8B 스냅샷은 부분 공개** — Wiki 물질화 태스크 하나가 신뢰도 숫자-문자열 변환에서 실패했고, `manifest.json`에 `release_status: incomplete_accepted_as_is`로 남겨뒀다
- **언어 차원은 캘리브레이션되지 않았다** — 다국어 사용자 분포에 의존하는 평가에는 그대로 못 쓴다
- **인프라 요구가 가볍지 않다** — 전체 파이프라인 재현은 SLURM 클러스터와 H200급 GPU, 수 TB 스토리지를 전제한다. 인덱스 파일(`postings.sqlite`)만 2.5GB다
- **웹·OS 앱 태스크는 Docker 이미지 빌드**가 필요해 첫 실행에 30~60분이 걸린다고 문서가 미리 경고한다

## 메모 — 내 관심사와의 접점

- **에이전트 평가 설계에 바로 꽂힌다** — 지금 하는 RAG·에이전트 평가는 골든셋 몇십 개에 LLM-as-judge를 얹는 구조다. 여기에 페르소나 축을 하나 더 세우면 "누가 물었을 때 답이 무너지는가"를 볼 수 있다. 도메인 지식 수준·기술 숙련도 차원만 골라 코호트를 잘라도 질의 난이도가 자연스럽게 층화된다
- **정직한 문서화가 인상적** — infeasibility를 클리핑하고 끝내는 게 아니라 클리핑된 질량을 리포트하고, 부분 공개를 부분 공개라고 매니페스트에 적고, 캘리브레이션 근거의 신뢰도를 차원별로 구분해 표기한다. 우리 평가 명세서도 이 수준으로 "무엇을 검증하지 않았는가"를 남겨야 한다
- **합성 데이터 dedup 기법이 재사용 가능** — 고차원 범주형 벡터에서 MinHash 대신 엔트로피 상위 필드를 좌표 투영하고 HyperLogLog로 폭을 정한다. 이 접근은 우리가 청크·질의 중복을 걸러낼 때도 그대로 응용된다. 전체 유사도 계산 없이 다양성 버킷만 만들면 되는 상황이 꽤 있다
- **결정적 지수 경주 샘플링** — 시드와 안정적 행 ID만으로 입력 순서에 무관하게 정확히 n개를 뽑는다. 실험 재현성이 걸린 데이터셋 샘플링에 그대로 쓸 수 있는 레시피다
- **경계할 점** — 페르소나가 그럴듯한 응답을 만든다고 그게 실제 사용자 분포를 대변하지는 않는다. LLM이 페르소나를 연기할 때 생기는 고유의 편향(모델 자체의 문체·성향)이 결과에 섞여 들어간다. 이 레포도 그 부분까지는 해결하지 않는다. 실사용자 데이터와 교차 검증하는 단계가 따로 필요하다

## 참조

| 주제 | 링크 | 활용 |
|------|------|------|
| MatrAIx 레포 | https://github.com/MatrAIx-ai/MatrAIx-Persona-8B | 소스·README·태스크 예제 |
| MatrAIx 핸드북 | https://github.com/MatrAIx-ai/MatrAIx-Persona-8B/tree/main/docs | 퀵스타트·설정·환경 문서 |
| 페르소나 파이프라인 | https://github.com/MatrAIx-ai/MatrAIx-Persona-8B/blob/main/docs/persona/pipeline.md | 스키마·추출·합성·후처리 상세 |
| Persona 1M 데이터셋 | https://huggingface.co/datasets/MatrAIx2026/MatrAIx_Persona_1M_Public_Release | 공개 코어셋 100만 행 |
| Persona8B 스냅샷 | https://huggingface.co/MatrAIx | 84억 행 통합 코퍼스 미러 |
| 공식 사이트 | https://matraix.ai/ | 프로젝트 소개·데모 |
