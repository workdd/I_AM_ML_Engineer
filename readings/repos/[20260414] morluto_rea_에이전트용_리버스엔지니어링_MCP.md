# REA: 에이전트에 붙이는 리버스 엔지니어링 MCP

- **레포**: [morluto/rea](https://github.com/morluto/rea) · [웹사이트](https://rea.tools/) · [npm `rea-agents`](https://www.npmjs.com/package/rea-agents)
- **제작**: morluto (기여자 30명 · 2026-10-16 기준)
- **공개**: 2026-04-14 저장소 생성일 (TypeScript, MIT, 별 269,618개 · 2026-10-16 GitHub API 기준)
- **읽은 날짜**: 2026-10-16 (커밋 `d52a488`, 릴리즈 `rea-agents v6.5.0`)
- **태그**: #MCP #ReverseEngineering #Ghidra #Electron #Evidence #PromptInjection

## 한 줄 요약

에이전트가 소스 없는 앱(네이티브 바이너리, Electron, .NET, APK, 웹사이트)을 MCP로 뜯어보고, 결론마다 **근거(Evidence)와 모르는 부분(Unknown)을 따로 기록해** 돌려주는 리버스 엔지니어링 도구 모음이다. 분석 엔진을 새로 만든 게 아니라 Hopper, Ghidra, IDA, JADX 같은 기존 도구를 한 MCP 계약 아래로 묶은 어댑터다.

## 쓰는 흐름

```
npx rea-agents setup        에이전트(Claude Code, Codex, Cursor, Gemini CLI 등)에 MCP 서버 + 작업 지침(skill) 등록
        │
에이전트에게 질문          "Notes 앱의 검색이 어떻게 동작하는지 근거와 함께 보여 주고, 내 프로젝트에 비슷하게 만들어 줘"
        │
REA 도구 호출 (로컬 실행)   대상 검사 → 관련 코드 추적 → Evidence + Unknown 반환
        │
에이전트                   설명, 후속 질문, 구현과 테스트
```

같은 기능을 CLI(`rea analyze-javascript-application <경로> --json` 등)로도 쓴다. README는 "기능을 보고 바이너리 수준까지 이해한다"를 내세우고, 사례도 기능 재현 쪽이다.

## 분석 대상

README의 대상 표에서 주요한 것만 골랐다.

| 대상 | 돌려주는 것 | 필요한 것 |
|---|---|---|
| 네이티브 바이너리 | 의사코드, 어셈블리, 문자열, 심볼, 호출·참조 | Hopper, Ghidra, IDA 중 하나 (objdump, Rizin도 가능) |
| JavaScript / Electron | 모듈, import, 소스맵, 라우트, IPC, 네이티브 애드온 관계 | Node.js만 (실행하지 않고 파일만 읽음) |
| 웹사이트 | 페이지 구조, 스크립트, 네트워크 관측, 스크린샷 | Chrome 계열 브라우저 |
| .NET 어셈블리 | 메타데이터, CIL 명령, 네이티브 의존성 | 없음 (정적 분석) |
| Android APK / 기기 | 매니페스트, 디컴파일 메서드 / 패키지, 로그, 화면 | JADX + JDK / adb |
| Go · Rust 바이너리 | 컴파일러 버전, 모듈·크레이트 의존성 | 없음 (내장 정적 분석) |
| 그 밖의 대상 | WASM, EVM 바이트코드, SQLite, 펌웨어, pcap·HAR, 크래시 덤프, 로직 분석기 기록, 프로세스 동작 | 대상별 외부 도구 |

대부분 Linux x64 기준으로 검증돼 있고, macOS·Windows 지원은 대상과 엔진마다 다르다.

## 구조

`docs/architecture.mermaid` 기준이다. 코드 규모는 `src/` 아래 TypeScript 파일 1,966개, 약 46만 줄이다(테스트 포함).

```
진입점      CLI (src/cli.ts) · MCP stdio 서버 (src/main.ts)
              │  같은 워크플로를 공유
애플리케이션   src/application/  세션, 엔진 선택(AnalysisProviderRegistry), 조사 기록(Evidence 원장, Unknown 등록부)
              │
엔진 어댑터   hopper/ (Hopper 안에서 도는 Python 브리지) · ghidra/ (Java 헤드리스 스크립트) · ida/ (ida-pro-mcp 연동)
            browser/ (CDP·Playwright) · android/ (JADX) · dotnet/ · firmware/ · pcap/ · gdb/ · rizin/ ...
              │
프로세스 기반  src/process/  외부 도구를 자기 소유 프로세스 그룹으로 띄우고, 시간 제한·읽기 한도·정리를 강제
```

- **대상 하나에 엔진 하나를 고정한다.** 엔진 선택은 정해진 순서대로 하고, 실패해도 다른 엔진으로 넘어가지 않는다(`deterministic selection, no fallback`). 같은 질문에 엔진이 바뀌어 답이 달라지는 일을 막는다.
- **외부 도구는 전부 감싸서 띄운다.** 시간 제한, 출력 크기 제한, 임시 폴더 정리를 REA가 맡는다.

## 눈여겨볼 설계

### 근거(Evidence)와 모르는 부분(Unknown)을 분리한다

`docs/mcp-contracts.md` 기준이다.

- 분석 도구는 결과마다 `evidence_id`가 붙은 Evidence 기록을 돌려준다. 세션 안에 원장으로 쌓이고, 묶음(bundle)으로 내보내 다른 세션이나 CLI에서 다시 읽을 수 있다.
- 결과에는 `coverage.status`(`complete` 또는 `unknown`)가 따로 붙는다. 그래프 탐색이 잘렸으면 `truncated`, 호출 경로 탐색이 다 못 돌았으면 `residual_unknowns`에 남긴다. **"못 찾았다"와 "없다"를 구분한다.**
- 풀리지 않은 질문은 Unknown 등록부(`record_unknown`, `list_unknowns`, `update_unknown`)에 남고, MCP 프롬프트 `audit_residual_unknowns`로 남은 것을 점검한다.

에이전트가 그럴듯하게 메워 넣는 것을 계약 수준에서 막는 방식이라, RAG 답변의 근거 표시나 "근거 없음" 처리에도 그대로 가져다 쓸 만하다.

### 분석 대상의 텍스트를 신뢰하지 않는다

바이너리 문자열, 디컴파일 코드, 페이지 글, 로그는 분석 대상이 만든 것이라 에이전트를 노린 지시문이 숨어 있을 수 있다. REA는 이 문제를 이렇게 다룬다.

- 대상 내용을 돌려줄 수 있는 도구의 결과에는 성공·실패와 관계없이 `_meta.provenance = { untrusted_content: true, source: "analysis_target" }` 표식을 붙인다.
- 표식은 기본으로 켜져 있다. 세션 상태만 돌려주는 도구(`export_evidence_bundle`, `record_unknown`)만 감사를 거쳐 뺀다. 새 도구는 감사 전까지 표식이 붙는다.
- 서버 지침은 에이전트에게 결과 안의 지시를 따르지 말고, 실행·설치·다운로드·전송 요청이 보이면 프롬프트 인젝션 시도로 보고하라고 한다.
- 호스트나 훅이 이 표식을 보고, 대상 내용을 읽은 뒤에는 프로세스 실행 도구에 승인을 요구하는 식으로 더 엄격하게 처리할 수 있다.

MCP 서버를 만들 때 도구 결과에 출처 표식을 거는 좋은 예다.

### 응답 크기 한도

결과가 응답 한도를 넘으면 잘라 버리지 않고 요약 Evidence와 이어 읽기 참조(`continuation`)를 준다. 전체는 `inspect_analysis_view`로 나눠 보거나 파일로 내보낸다.

## 한계와 주의

- **엔진 라이선스는 사용자 몫이다.** Hopper, IDA, JEB는 상용이고, REA는 키나 라이선스를 주지 않는다고 README에 명시돼 있다. 무료로 깊게 보려면 Ghidra다.
- **사용 범위 제한.** `DISCLAIMER.md`는 DRM·라이선스 검사 우회, 치트, 인증 우회, 독점 코드 복제를 지원하지 않는다고 밝힌다. "기능을 보고 내 제품에 만든다"는 홍보 문구와 "보호된 코드를 복제하지 않는다"는 경계가 가깝다. 남의 앱을 분석할 때는 라이선스 약관부터 확인해야 한다.
- **분석은 로컬이지만 결과는 모델로 간다.** 도구 결과는 에이전트의 모델 공급자에게 전달되므로, 사내 바이너리를 분석하면 그 내용이 외부 API로 나간다.
- **변경이 빠르다.** 6.4.0 → 6.5.0이 3일 간격이고, README도 최신 버전 유지를 먼저 권한다. 계약 이전 안내(`migration-6.2.md`)가 따로 있을 만큼 결과 형식도 바뀐다.
- **별 수는 걸러 볼 것.** 생성 반년 만에 별 26만 개, 기여자 30명이다. 인기의 지표로 그대로 받기는 어렵다(미검증).
- 도구 개수는 직접 세지 못했다. 계약 스키마에 도구 이름 꼴 문자열이 126개 있지만 전부 도구인지는 확인하지 않았다(미검증). 정확한 목록은 `npm run docs:generate`로 만드는 카탈로그나 `binary_session` 도구에서 본다.

## 가져올 수 있는 부분

| 대상 | 가져올 수 있는 부분 |
|---|---|
| 에이전트·MCP 서버 설계 | 대상 유래 텍스트에 `untrusted_content` 표식을 기본으로 켜고, 감사한 도구만 빼는 방식 |
| RAG 답변 근거 | 결과마다 coverage(`complete`/`unknown`)와 남은 Unknown을 따로 돌려주는 계약 |
| 큰 도구 결과 | 자르지 않고 요약 + 이어 읽기 참조로 나누기 |
| 외부 도구 연동 | 엔진을 정해진 순서로 하나만 고르고 넘어가지 않기 (재현성) |
