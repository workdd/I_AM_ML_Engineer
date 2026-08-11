# late.sh — SSH 한 줄로 들어가는 터미널 클럽하우스

- **레포**: [mpiorowski/late-sh](https://github.com/mpiorowski/late-sh) · [late.sh](https://late.sh)
- **제작**: Mateusz Piórowski (단독 메인테이너 · 커밋 527/713, 컨트리뷰터 36명)
- **공개일**: 2026-04-11 (Rust, FSL-1.1-MIT, ⭐1,339 · 2026-08-11 기준)
- **읽은 날짜**: 2026-08-11
- **태그**: #SSH #TUI #ratatui #russh #Rust #LLMFirstDocs #Kubernetes #DoorGame

## 핵심 문제의식

`ssh late.sh` 한 줄이면 계정 생성도 앱 설치도 없이 채팅·음악·게임·뉴스가 있는 공용 공간에 들어간다. 로그인 이름은 버려지고 SSH 키 지문으로 계정을 찾거나 만든다. 회원가입 폼이 사라진 자리를 SSH 공개키가 대신하는 구조다.

BBS 시절 도어 게임 문화를 지금 스택으로 되살린 프로젝트인데, 재미 요소보다 **엔지니어링 밀도**가 볼 만하다. 러스트 워크스페이스 11개 크레이트에 34.7만 줄, 마이그레이션 162개, 단일 RKE2 노드에서 실제 트래픽을 받는다. 개인 프로젝트가 어디까지 갈 수 있는지 보여주는 표본에 가깝다.

## 설계 축

| 축 | 방법 | 메모 |
|----|------|------|
| 접속 장벽 제거 | russh SSH 서버(2222)가 곧 애플리케이션 진입점 | 키 지문 → 계정 자동 생성, 첫 접속 시 랜덤 유저네임 부여 |
| 렌더 예산 관리 | ratatui TUI를 15fps(66ms tick) 동기 루프로 고정 | 동시 세션이 늘어도 프레임 비용이 선형으로만 늘도록 |
| 동기/비동기 경계 | `state.rs`(동기)·`svc.rs`(비동기)를 채널로만 연결 | 상태 코드에는 `.await`를 아예 두지 않는다 |
| 외부 게임 흡수 | 도어 게임을 별도 크레이트 + PTY 호스트로 격리 | NetHack·DCSS·Brogue·dopewars·Usurper는 진짜 업스트림 바이너리 |
| 문서 우선 | 저장소 전역에 `CONTEXT.md` 27개 배치 | LLM 에이전트를 1차 독자로 명시한 문서 체계 |

## 구조

```
사용자 터미널 ──SSH(2222)──► late-ssh (russh + ratatui TUI)
                                 │   ├── SessionRegistry     (token → mpsc)
                                 │   └── PairedClientRegistry(token → WS + state)
                                 │
                                 ├──► PostgreSQL (CloudNativePG, 마이그레이션 162개)
                                 ├──► HTTP API (axum :4000) ◄──WS── late CLI (로컬 오디오·비주얼라이저)
                                 └──► LiveKit (음성 룸)

브라우저 ──► late-web (axum :3000) ──► /listen · 프로필 · /stream 프록시
                                            │
                            Liquidsoap ──► Icecast (오디오 스트림)

도어 게임 호스트 (독립 크레이트, 각자 PTY):
  late-nethack · late-dcss · late-brogue · late-dopewars · late-usurper · late-codekeep
```

크레이트별 규모를 보면 무게중심이 확실하다.

| 크레이트 | 줄 수 | 역할 |
|---|---|---|
| `late-ssh` | 297,776 | SSH 서버 + TUI 전체. 화면·게임·서비스가 모두 여기 |
| `late-core` | 26,444 | 공유 도메인, DB 계층, 마이그레이션 |
| `late-cli` | 7,199 | 동반 CLI — 로컬 오디오 재생, 페어링, 비주얼라이저 |
| `late-web` | 1,723 | 랜딩·프로필·`/listen` |
| 도어 호스트 6종 | 각 750~1,700 | 업스트림 바이너리를 PTY에 태우는 얇은 래퍼 |

### 도메인 모듈 규칙

`late-ssh/src/app/<domain>/` 아래는 파일 역할이 고정돼 있다. 재export 없는 평평한 구조다.

```
mod.rs     # pub mod 선언만, pub use 재export 금지
state.rs   # 동기 UI 상태. tick마다 채널을 비우고 갱신, .await 없음
input.rs   # 키 라우팅. I/O가 필요하면 서비스에 fire-and-forget 호출
ui.rs      # 순수 ratatui 그리기 함수
svc.rs     # 비동기 서비스 — DB, 브로드캐스트, 백그라운드 태스크
model.rs   # DB 기반 타입
```

입력이 상태를 즉시 바꾸지 않고 서비스에 던진 뒤 결과가 다음 tick의 채널로 돌아온다. 렌더 루프가 DB를 기다리며 멈추는 일이 아예 없다.

### 도어 게임 통합 3패턴

1. **네이티브 러스트 포팅** — Lateania, Green Dragon. 공수가 가장 크지만 완전한 통제권을 얻는다. 게임 메커니즘 자체에는 저작권이 없으므로 재구현은 라이선스에 걸리지 않는다
2. **업스트림 바이너리를 PTY에 올려 SSH로 프록시** — NetHack, DCSS, Brogue, dopewars, Usurper. 이미 유닉스 터미널 프로그램인 게임은 이쪽이 압도적으로 싸다
3. **원격 SSH 도어 프록시** — Rebels in the Sky. 이미 SSH/telnet 서버를 노출하는 게임

`DOOR.md`에 후보 게임을 라이선스 신호등으로 분류해둔 대목이 눈에 띈다. TradeWars 2002는 상표권 때문에 탈락, LORD는 라이선스가 막혀 오픈소스 리메이크인 LotGD로 우회하는 식으로 **법적 검토가 기술 검토보다 앞선다.**

## 사용법

```bash
# 라이브 서비스 접속
ssh late.sh

# 로컬 실행 (Docker 필요) — Postgres·Icecast·Liquidsoap까지 함께 뜬다
git clone https://github.com/mpiorowski/late-sh
cd late-sh && make start
ssh localhost -p 2222

# 인프라만 도커, 앱은 네이티브
docker compose up -d postgres icecast liquidsoap
cargo run -p late-ssh
cargo run -p late-web

# PR 전 로컬 게이트 (fmt → clippy → nextest, --features otel)
make check
```

동반 CLI(`late`)를 깔면 오디오가 서버 스트림이 아니라 로컬에서 재생되고 비주얼라이저가 TUI와 동기화된다. CLI는 WebSocket으로 API에 붙어 페어링 토큰을 주고받는다.

## LLM을 1차 독자로 삼은 문서 체계

이 레포에서 가장 가져올 만한 부분이다. 루트 `CONTEXT.md` 1,309줄을 포함해 저장소 전역에 `CONTEXT.md`가 27개 있고 README가 대놓고 "LLM용으로 썼으니 AI 에디터에 먹이라"고 안내한다.

- **Read-First 라우팅 표** — 루트 문서 상단에 "이 도메인을 건드리면 이 문서부터 읽어라" 매핑이 표로 있다. 작업이 도메인을 넘나들면 해당 행을 전부 읽고 루트와 로컬 문서를 함께 갱신하도록 규정
- **`[STABLE]` / `[VOLATILE]` 태그** — 섹션마다 변경 빈도를 표시해서 자주 바뀌는 곳과 계약에 해당하는 곳을 구분
- **메타데이터에 마지막 변경 사유를 통째로 기록** — 날짜만 적는 게 아니라 "무엇이 왜 바뀌었는지"를 몇 문장으로 남긴다
- **유지 프로토콜을 문서 0번 절에 명시** — 코드와 문서가 어긋나면 문서를 먼저 고치라고 못 박음
- **인시던트 로그·성능 노트 분리** — 용량·성능 발견은 `SCALE.md`로 몰아두고 루트에는 현재 계약만 남긴다

문서를 컨텍스트 윈도우에 넣을 자산으로 취급하고 라우팅·신선도·안정성 메타데이터를 붙인 사례다. 사내 레포에 그대로 적용해볼 만하다.

## 운영 현실 (SCALE.md)

- 클러스터는 **RKE2 단일 노드** 하나. 8 CPU / 15.6 GiB
- 동시 60세션에서 CPU 37%, 메모리 46% (렌더 비용 최적화 이후). 최적화 전에는 80세션에 CPU 77%였다
- 목표는 동시 1,000명. `LATE_MAX_CONNS_GLOBAL`은 이미 1000으로 올려둠
- Hacker News 유입 스파이크로 두 번 사고(2026-06, 2026-07-22 OOM). 그때 `service-ssh` 메모리 한도를 4→8 GiB로 올리고 CPU 한도를 4→8코어로 상향
- `termination_grace_period_seconds`가 **21600초(6시간)**. 배포해도 기존 SSH 세션이 끊기지 않게 구 파드를 최대 6시간 살려둔다
- 모든 PVC가 `local-path` 프로비저너라 볼륨 있는 파드는 노드에 고정된다. 노드 증설 시 도어 게임 세이브·음악 데이터·Postgres가 걸림돌

인프라는 Terraform 5,233줄로 관리한다. PostgreSQL은 CloudNativePG 2인스턴스다.

## AI 기능

`late-ssh/src/app/ai/`에서 **Gemini 3.6 Flash** 하나로 처리한다.

- **채팅 번역** — 메시지 선택 후 `t`, 새 메시지 자동 번역 옵션. 결과를 캐시해 여러 사용자가 공유한다. 모델이 판정한 `same_language` 결과까지 캐시해서 같은 언어 감지에 매번 호출하지 않는다
- **링크·유튜브 요약** — 공유한 링크에 요약과 ASCII 썸네일을 붙인다
- **Ghost(바텐더 NPC)** — 라운지 상주 캐릭터

## 제약

- **FSL-1.1-MIT — 오픈소스가 아니다.** 읽기·수정·사내 사용·비영리 연구·PR은 허용, **경쟁하는 공개 호스팅 서비스 운영은 금지**. 공개 2년 뒤 MIT로 자동 전환된다(2028-04-11 전후). GitHub는 이 라이선스를 `NOASSERTION`으로 표기
- 브랜딩 사용 금지 — 포크를 공식 `late.sh`처럼 내세울 수 없다
- 커밋에 DCO 사인오프(`git commit -s`) 필수
- `late-ssh` 한 크레이트가 29.8만 줄이라 부분만 떼어 참고하기는 어렵다. 빌드도 무거워서 개발 프로파일 디버그 정보를 `line-tables-only`로 낮춰놨다(스왑 없는 머신에서 `cargo nextest`가 얼어붙어서)
- 단일 노드 · 단일 레플리카 구조라 무중단 운영이 아니라 "6시간짜리 유예"로 버틴다. 멀티 레플리카 대응은 로드맵 상태

## 메모 — 내 관심사와의 접점

- **CONTEXT.md 체계를 그대로 훔쳐올 만하다.** 지금 CLAUDE.md 하나로 버티는 구조보다 도메인별 컨텍스트 파일 + Read-First 라우팅 표 + `[STABLE]`/`[VOLATILE]` 태그 조합이 명백히 낫다. 특히 라우팅 표는 에이전트가 무엇을 읽어야 할지 스스로 결정하게 해준다
- **번역 캐시 설계가 참고할 만하다.** 판정 결과(`same_language`)까지 캐시해서 불필요한 모델 호출을 잘라내는 건 LLM 비용 최적화의 기본기다. "번역 안 해도 되는 케이스"를 모델에게 한 번 물어보고 영구 저장하는 방식
- **동기 렌더 루프와 비동기 서비스의 분리**는 스트리밍 LLM UI에 그대로 옮길 수 있는 패턴이다. 상태는 동기로 두고 모델 응답은 채널로 흘려보내면, 토큰이 늦게 와도 UI가 멈추지 않는다
- **개인 프로젝트의 스케일 상한선 관찰용 표본.** 커밋의 74%가 한 사람 손에서 나왔고 4개월 만에 34.7만 줄에 별 1.3k, 실서비스 운영까지 왔다. LLM 에이전트를 전제로 문서를 짠 게 이 속도와 무관하지 않아 보인다
