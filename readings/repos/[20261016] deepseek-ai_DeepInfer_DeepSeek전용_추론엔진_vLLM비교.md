# DeepInfer: DeepSeek가 공개한 자사 서비스용 추론 엔진과 vLLM 비교

- **레포**: [deepseek-ai/DeepInfer](https://github.com/deepseek-ai/DeepInfer) · [배포 Q&A 문서](https://github.com/deepseek-ai/DeepInfer/blob/main/doc/inference_deployment_qa.md) · [평가 README](https://github.com/deepseek-ai/DeepInfer/blob/main/eval/README.md)
- **제작**: DeepSeek-AI (기여자 3명, 익명 포함 · 2026-10-23 GitHub API 기준)
- **공개**: 2026-10-16 저장소 생성일 (Python 약 2.3MB, C++ 약 0.28MB, Apache-2.0, 별 20,208개 · 2026-10-23 기준)
- **읽은 날짜**: 2026-10-23 (커밋 `9bd9cf2`)
- **원형**: README의 Acknowledgments에 "originally built primarily on vLLM"이라고 적혀 있고, SGLang과 TensorRT-LLM도 참고했다고 밝힌다
- **비교 대상**: [vllm-project/vllm](https://github.com/vllm-project/vllm) README (별 95,188개 · 2026-10-23 기준)
- **태그**: #Inference #vLLM #DeepSeek #MoE #ExpertParallel #PDDisaggregation #SpeculativeDecoding #KVCache

## 한 줄 요약

DeepSeek-V4.1-Flash 서비스를 실제로 돌리는 엔진에서 **오프라인 추론 부분만** 떼어 공개한 것이다. vLLM에서 출발했지만 모델 하나와 하드웨어 둘(NVIDIA SM100, Huawei Ascend 950)에만 맞춰 깎았다. 코드보다 더 쓸모 있는 것은 함께 공개한 **배포 Q&A 문서**다. 대규모 MoE를 prefill/decode 분리 구조로 운영하는 실제 설정값이 들어 있다.

## vLLM과 비교

| 항목 | DeepInfer | vLLM |
|---|---|---|
| 성격 | 자사 프로덕션 엔진의 일부 공개 | 범용 오픈소스 엔진 |
| 지원 모델 | DeepSeek-V4.1-Flash 하나 | Hugging Face 모델 구조 200종 이상 |
| 하드웨어 | NVIDIA SM100, Huawei Ascend 950 | NVIDIA, AMD, Intel GPU, CPU. 플러그인으로 TPU, Gaudi, Ascend 등 |
| 제공 형태 | 오프라인 배치 API만 (`deepinfer/entrypoints/llm.py`) | OpenAI 호환 API 서버, Anthropic Messages API, gRPC |
| P/D 분리 | 엔진 기능은 대부분 구현했지만 이번 공개본에는 없음 | prefill, decode, encode 분리 지원 |
| KV 캐시 시스템 | 공개본에 없음 (사내 3FS 기반 저장소에 묶여 있음) | PagedAttention, prefix caching |
| 병렬화 | Attention DP + MoE EP + EPLB. 운영에서 TP와 CP는 쓰지 않음 | TP, PP, DP, EP, CP |
| 투기적 디코딩 | DSpark (draft 신뢰도로 검증 길이를 정함) | n-gram, suffix, EAGLE, DFlash |
| 구조화 출력 | regex, JSON schema, EBNF | xgrammar, guidance |
| 결정성 | batch-invariant 추론, 결정적 샘플링 (NVIDIA만) | README에 별도 항목 없음 |
| 최소 효율 배포 | prefill 64 GPU + decode 64 GPU | GPU 1장부터 |

vLLM 쪽 항목은 vLLM README의 기능 목록을 기준으로 했고 코드까지 확인하지는 않았다.

## 공개된 것과 빠진 것

README가 "완전한 서비스에 필요하지만 이번 공개본에 없는 것"으로 직접 적은 세 가지다.

- KV 캐시 시스템
- Vision Encoder / Prefill / Decode 분리 배포 (엔진 쪽 기능은 대부분 구현돼 있다고 함)
- prefill과 decode 양쪽의 DP 복제본 간 로드밸런싱

배포 Q&A 문서는 이 두 부품(KV 캐시 관리, 로드밸런서)이 사내 인프라에 묶여 있어 아직 공개하지 못했다고 설명한다. 대신 설계는 문서와 Zhihu 블로그로 공개했다. API 프로토콜과 토큰 사이의 변환은 별도 레포 [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe)로 공개했다.

## 엔진 기능

| 기능 | 내용 | 켜는 방법 |
|---|---|---|
| 커널 | FlashMLA, DeepGEMM, DeepEP, TileKernels, DeepSelect, 자체 TileLang 커널 | 기본 |
| DSpark | draft 토큰마다 Confidence Head로 수용 확률을 예측하고, 기대 수용률 순으로 정렬해 처리량이 가장 큰 앞부분만 검증 | `DEEPINFER_ENABLE_MTP_VARLEN=1`, 최대 draft 길이 `DEEPINFER_USE_NEXTN_DECODE` (기본 5) |
| SWA bounded replay | decoder prefill에서 끝부분만 다시 계산 | `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, `DEEPINFER_YOCO_PREFILL_KEEP_TOKENS=140` |
| 비동기 스케줄링 | GPU가 `forward(N)`을 도는 동안 CPU가 `schedule(N+1)`과 `post-process(N-1)`을 처리해 GPU 유휴 구간을 없앰. SGLang, NanoFlow 방식을 따름 | 기본으로 켜짐 |
| Batch-invariant | 배치 구성이 달라도 결과가 같음 | `DEEPINFER_BATCH_INVARIANT=1` (NVIDIA만) |
| 결정적 샘플링 | 같은 입력과 샘플링 설정이면 같은 출력 | 위 설정 + `DEEPINFER_ENABLE_MTP_VARLEN=0` + `SamplingParams`에 `seq_seed` 고정 |
| 구조화 출력 | 별도 SO 머신에서 문법 오토마톤을 비동기로 컴파일하고, decode 단계마다 마스크 적용 | `SamplingParams(so_args=...)` |

설치 조건은 Linux, GCC 13 이상, Python 3.12, CUDA 13(NVIDIA) 또는 CANN 9.2(Ascend)다.

## 배포 Q&A 문서에서 나온 운영 수치

`doc/inference_deployment_qa.md` 기준이다. 모두 DeepSeek의 자체 발표이고 직접 재현하지 않았다.

### 시스템 구조

- Vision Encoder / Prefill / Decode를 분리하고 KV 캐시 전용 저장 노드를 둔다
- prefill: EP2~8 / DP2~8. HBM이 허락하는 가장 작은 값을 쓴다
- decode: EP64~128 / DP64~128 (Wide-EP). 가용 GPU가 허락하는 가장 큰 값을 쓴다
- vision encoder: 인스턴스당 GPU 1장, 싼 GPU를 권장
- P와 D의 GPU 비율은 대략 1:1
- 클러스터 하나가 하루 155조 토큰을 처리하고, GPU 1시간당 매출은 38달러(비수기 가격 기준)

### 로드밸런싱

| 대상 | 방식 |
|---|---|
| Expert (prefill) | GPU당 중복 expert 4개, 100스텝 통계, 1,000스텝마다 재배치. `moe_lb`(가장 느린 rank / 평균) 1.074 (EP4DP4), 1.045 (EP2DP2) |
| Expert (decode) | GPU당 중복 expert 1개, 1,000스텝 통계, 10,000스텝마다 재배치. `moe_lb` 1.090 |
| Prefill | 문맥 길이로 4구간(<16K, 16~64K, 64~256K, ≥256K) 분리. GPU 안에서는 2차 비용 모델로 청크 크기를 시간 예산에 맞춰 풂 |
| Decode | 요청 수와 KV 캐시 총길이를 함께 맞춤. 짧은 요청은 동시 처리 수가 적고 KV가 긴 GPU로, 긴 요청은 KV가 짧은 GPU로 |
| 전체 효과 | 완전히 균형 잡힌 오프라인 벤치마크의 약 80% 수준 |

### KV 캐시 저장

| 내용 | 위치 |
|---|---|
| 처리 중인 block KV + state KV | 추론 GPU 메모리 |
| SWA KV | 로컬 호스트 메모리, 단일 사본. 다른 머신 히트는 400G VLAN으로 가져옴 |
| Paged block KV | 전용 3FS 저장 클러스터 (10+2 erasure coding) |

- 3FS는 KV 캐시용으로 약 100 PiB SSD를 쓰고, TTL은 에이전트 트래픽 3일, 챗봇 트래픽 7일이다
- 챗봇은 소수 KV가 아주 많이 다시 읽히는 긴 꼬리가 있어, TTL을 3일로 줄이면 히트율이 0.5~1% 떨어진다고 한다
- 저장 노드 하나에 30.73TB SSD 20개, 읽기 40GB/s 이상, 쓰기 5GB/s 이상을 권장한다
- 로컬 계층형 저장 대신 전용 클러스터를 쓰는 이유: 어느 prefill 머신이든 어떤 prefix에도 히트할 수 있어, prefix 친화 라우팅과 로드밸런싱이 충돌하지 않는다

### DSpark 동적 검증 길이 효과

EP64/DP64 벤치마크 기준으로, 고정 5개 검증 대비 검증 토큰 수가 약 30% 줄고(5.00에서 3.2~4.4) 평균 수용 길이는 거의 같다(3.29 대 3.20). 운영 지점(rank당 동시 요청 약 60개)에서 처리량이 4~7% 오른다. 표에서는 동시 요청 16개일 때 -1.5%, 96개일 때 +11.0%다.

### FP4 사용 위치

- Main KV: 16개 단위, e4m3 (전역 스케일 없는 NVFP4 형태)
- Indexer KV: 32개 단위, ue8m0, MXFP4
- MoE routed expert 가중치: 32개 단위, ue8m0, MXFP4

## 평가

`eval/`에 AIME 2026(30문항 × 32샘플), HLE(2,500문항, 텍스트 2,158 + 이미지 342), GPQA Diamond(198문항 × 8샘플)의 Pass@1 평가 스크립트가 있다. 생성과 채점을 나눠, 채점자를 바꿔도 다시 생성하지 않는다.

## 읽을 때 감안할 것

- **처리량 수치는 모두 자체 발표다.** README의 성능 그래프는 prefill 4 GPU, decode 64 GPU(SM100, 한 scale-up 도메인 안)에서 쟀고, 재현 스크립트(`examples/sweep/run_sweep.sh`)만 있다
- **바로 서빙할 수 없다.** HTTP 서버, KV 캐시 시스템, 로드밸런서가 없다. vLLM의 `vllm serve`에 해당하는 것이 없다
- **하드웨어 조건이 좁다.** Blackwell(SM100) 이전 세대 GPU는 지원 목록에 없다
- 공개 커널은 사내 버전과 거의 같지만, FlashMLA는 손으로 튜닝한 SASS 최적화 하나가 빠져 decode 커널이 5~10% 느리다고 문서가 직접 밝힌다. 종단 성능 차이는 5% 이내라고 한다
- 기여자가 3명으로 잡히는 것은 사내 저장소를 한 번에 옮겨 온 형태이기 때문으로 보인다(추측)

## 가져갈 지점

- **엔진 선택**: DeepSeek 모델을 자체 환경에서 돌려야 하면 지금은 vLLM이나 SGLang이 현실적이다. DeepInfer는 설정값과 구조를 참고하는 자료로 쓴다
- **P/D 분리와 Wide-EP 설계값**: prefill은 EP를 작게, decode는 크게 잡는 기준과 EPLB 통계 주기가 실제 운영값으로 나와 있다. [서빙 구조가 LLM 처리량을 바꾸는 방식](../blogs/[20260908]%20Notion_서빙구조가_LLM_처리량을_바꾸는_방식.md)과 같이 읽으면 좋다
- **KV 캐시를 전용 저장 클러스터에 두는 판단**: prefix 친화 라우팅을 포기하는 대신 어디서든 히트하게 만든 설계다. [LLM 서빙 성능최적화](../blogs/[20251219]%20네이버클로바_LLM서빙_성능최적화.md)의 KV 캐시 논의와 비교할 만하다
- **신뢰도 기반 투기적 디코딩**: 고정 길이 검증보다 동시 요청이 많을수록 이득이 커진다. [Speculative Decoding 적용기](../blogs/[20251219]%20네이버클로바_Speculative_Decoding_적용기.md)와 같은 문제를 다룬다
- **모델 쪽 배경**: KV 압축, SWA, YOCO 구조는 [DeepSeek-V4.1-Flash 기술 보고서 정리](../papers/[20260911]%20DeepSeek-V4.1-Flash_KV_Cache_Compression.md)에 있다

## 결론

DeepInfer는 vLLM을 대신할 엔진이 아니다. DeepSeek 모델 하나를 Blackwell 64장 이상 규모로 운영할 때의 정답지를 코드와 문서로 보여 주는 자료다. 엔진 코드보다 배포 Q&A 문서를 먼저 읽는 것이 맞다.
