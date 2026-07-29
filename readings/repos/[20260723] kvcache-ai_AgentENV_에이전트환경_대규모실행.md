# AgentENV — 에이전트 환경을 대규모로 굴리는 플랫폼

- **레포**: [kvcache-ai/AgentENV](https://github.com/kvcache-ai/AgentENV) · [공식 문서](https://kvcache-ai.github.io/AgentENV/)
- **제작**: kvcache-ai (Mooncake·KTransformers 조직)
- **공개일**: 2026-07-23 (Rust, MIT, ⭐1.5k · 2026-07-29 기준)
- **읽은 날짜**: 2026-07-29
- **태그**: #AgenticRL #Firecracker #microVM #Sandbox #E2B #overlaybd #KimiK3

## 핵심 문제의식

Agentic RL 학습은 롤아웃 하나마다 격리된 실행 환경(셸·파일시스템·툴)이 있어야 한다. 환경 수가 수만 개로 올라가면 병목은 모델이 아니라 인프라 쪽에서 먼저 터진다.

- **부팅 비용**: 롤아웃마다 컨테이너·VM을 새로 띄우면 시작 지연이 학습 처리량을 깎는다
- **이미지 배포**: 태스크마다 다른 OCI 이미지를 쓰는데, 클러스터 전 호스트에 미리 뿌려두는 방식은 디스크 용량에 막힌다
- **유휴 낭비**: 에이전트가 LLM 응답을 기다리는 동안 환경은 CPU·메모리를 잡은 채 놀고 있다

AgentENV(AENV)는 이 문제를 정면으로 다루는 플랫폼이다. README에는 **Kimi K3의 agentic RL 학습을 떠받친다**고 명시돼 있다.

## 설계 축 4가지

| 축 | 방법 | 수치 |
|----|------|------|
| 다양한 환경을 대규모로 | Firecracker microVM + OCI 이미지, overlaybd 온디맨드 로딩 | 로컬 디스크를 bounded cache로 써서 디스크 용량 초과 이미지도 pre-warm 없이 |
| 유휴 환경을 싸게 | 스냅샷 기반 부팅·재개, 유휴 시 CPU/메모리 반납 | 부팅·재개 <50ms, pause <100ms |
| 스냅샷·fork 네이티브 | 메모리+파일시스템 증분 스냅샷, 실행 중 환경 fork | 디스크 변경이 심해도 <100ms |
| 오래 돌려도 성능·밀도 유지 | ublk I/O, 호스트 page cache 공유, memory ballooning | 환경이 시간이 갈수록 갈라져도 overcommit 유지 |

로컬 디스크를 저장소가 아니라 캐시로 재정의한 게 핵심이다. hot 데이터는 남기고 cold는 축출하니, 이미지 총량이 디스크보다 커도 클러스터 전체 시작 속도가 유지된다.

## 구조

```
aenv CLI  /  E2B Python·TS SDK
      │  HTTP (E2B 호환 API)
      ▼
  API 서버 ──── 게이트웨이 + 스케줄러 (멀티노드, 프로토타입 단계)
      │
      ▼
  Firecracker microVM ◄── overlaybd + ublk (레이어드 블록 디바이스, CoW 공유)
      │                         │
      │                    로컬 디스크 = bounded cache (hot 유지 / cold 축출)
      │
      └── 스냅샷 (메모리 + FS) ──► S3 호환 오브젝트 스토리지 / 분산 파일시스템
```

## 사용법

```bash
# 설치 (Ubuntu 24.04) + 서버 기동
curl -fsSL https://raw.githubusercontent.com/kvcache-ai/AgentENV/main/scripts/install.sh | sudo bash
sudo systemctl start aenv          # 기본 http://127.0.0.1:8000

# 템플릿 pull → 샌드박스 실행
aenv pull ubuntu:22.04 --name ubuntu
aenv start ubuntu                  # 실행 + 인터랙티브 셸 attach
aenv start ubuntu --detach         # 실행 후 sandbox ID만 출력

aenv exec <sandbox-id> ls -la /    # 일회성 명령
aenv pause  <sandbox-id>
aenv resume <sandbox-id>
aenv timeout <sandbox-id> 600      # TTL 연장
```

Docker 배포와 소스 빌드도 지원한다. 클러스터는 Docker Compose·Kubernetes 문서가 따로 있다.

## E2B 호환

`E2B_API_URL`만 자기 서버로 돌리면 표준 E2B Python/TypeScript SDK를 코드 수정 없이 쓸 수 있다. E2B SaaS를 쓰던 자리를 self-host로 갈아끼운다는 얘기다. 실무에서는 이게 가장 큰 진입 장벽 제거다.

## 제약

- Linux 커널 **6.8 이상**, Firecracker 실행을 위한 `/dev/kvm` 접근 필요 (설치 스크립트는 Ubuntu 24.04 전제)
- **인증·인가 미지원** — README가 경고로 못 박는다. 공개망에 API를 노출하지 말고 신뢰 네트워크나 인증 프록시 뒤에서만 돌릴 것
- 멀티노드 게이트웨이·스케줄러는 문서상 프로토타입 단계

## 메모 — 내 관심사와의 접점

- **툴 실행 샌드박스 후보**: 멀티 에이전트·agentic RAG 실험에서 코드 실행을 격리할 때, 지금 E2B SaaS가 맡는 자리를 그대로 대체할 수 있다. API가 호환되니 전환 비용이 거의 없다
- **fork가 진짜 물건**: 실행 상태 하나에서 여러 갈래로 갈라 병렬 롤아웃을 돌릴 수 있다. Best-of-N이나 트리 탐색형 에이전트 평가에 바로 꽂히는 기능이다. 앞 단계 셋업을 매번 다시 밟지 않아도 된다
- **조직 맥락**: kvcache-ai는 Mooncake(Kimi 서빙 플랫폼)와 KTransformers를 내놓은 팀이다. 추론 인프라를 하던 쪽이 학습용 환경 인프라까지 손대는 흐름이다. agentic RL의 병목이 어디로 옮겨갔는지가 여기서 드러난다

## 참조

| 주제 | 링크 | 활용 |
|------|------|------|
| AgentENV 레포 | https://github.com/kvcache-ai/AgentENV | 소스·README·CLI |
| 공식 문서 | https://kvcache-ai.github.io/AgentENV/ | 배포·API·E2B 연동 |
| E2B 연동 | https://kvcache-ai.github.io/AgentENV/integration/e2b.html | SDK 호환 설정 |
| overlaybd | https://containerd.github.io/overlaybd/ | 온디맨드 이미지 로딩 기반 기술 |
| Mooncake | https://github.com/kvcache-ai/Mooncake | 동일 조직의 Kimi 서빙 플랫폼 |
