# DeepInfer 해부: DeepSeek 추론 엔진 코드 분석

DeepSeek가 2026년 10월 공개한 추론 엔진 [DeepInfer](https://github.com/deepseek-ai/DeepInfer)를 코드 단위로 따라가는 책이다. 엔진 루프, KV 캐시, DeepSeek-V4 어텐션, MoE expert 병렬, MTP 추측 디코딩, prefill/decode 분리, NVIDIA와 Ascend 이원화까지 실제 소스 위치를 링크로 걸어 설명한다.

- 기준 커밋: [`a679bb8`](https://github.com/deepseek-ai/DeepInfer/tree/a679bb86ce36e883a98a97733492b686e5b9a08e) (2026-11-03)
- 원본 라이선스: Apache-2.0. 코드 인용은 짧게 하고 원문 링크를 함께 건다
- 원고 저장소: [workdd/I_AM_ML_Engineer](https://github.com/workdd/I_AM_ML_Engineer) `books/deepinfer`
