설치 경로는 릴리스 바이너리를 내려받는 방법과 소스에서 직접 빌드하는 방법 두 가지다.

## 릴리스 바이너리

README 는 Releases 페이지에서 macOS(Apple Silicon) `.dmg`, Windows `.msi`, Linux `.deb` / `.AppImage` 를 받으라고 적는다. 실제로 무엇이 올라가는지는 릴리스 워크플로가 정한다.

### 코드 분석: 릴리스 빌드 행렬

빌드 대상 플랫폼 목록을 본다. [.github/workflows/build.yml L3-L31](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L3-L31)

```yaml
on:
  push:
    tags:
      - 'v*'
  workflow_dispatch:
...
        include:
          - platform: macos-latest
            args: '--target aarch64-apple-darwin'
            rust_target: aarch64-apple-darwin
          - platform: ubuntu-22.04
          - platform: ubuntu-22.04-arm
          - platform: windows-latest
```

`v*` 태그를 push 하면 네 플랫폼에서 빌드가 돈다. README 에는 없지만 행렬에 `ubuntu-22.04-arm` 이 있어서 Linux ARM64 번들도 만들어진다. macOS 는 `aarch64-apple-darwin` 하나만 지정하므로 Intel Mac 용 번들은 이 워크플로에서 나오지 않는다.

같은 워크플로의 뒷부분을 보면 산출물이 README 목록보다 넓다.

| 단계 | 하는 일 | 위치 |
|---|---|---|
| PDFium 체크섬 검증 | 저장소에 커밋된 PDFium 바이너리를 `SHA256SUMS` 로 확인 | [build.yml L80-L90](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L80-L90) |
| MCP 서버 준비 | `npm --prefix mcp-server ci` 후 `npm run mcp:build` | [build.yml L106-L109](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L106-L109) |
| Tauri 빌드 | `tauri-action` 으로 번들 생성, 태그 빌드는 draft 릴리스 | [build.yml L111-L134](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L111-L134) |
| macOS 서명 검증 | 서명 · 공증 결과 확인 스크립트 실행 | [build.yml L136-L138](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L136-L138) |
| Windows 포터블 zip | 설치 없이 쓰는 zip 을 만들어 릴리스에 첨부 | [build.yml L140-L153](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L140-L153) |
| 수동 실행 산출물 | `workflow_dispatch` 일 때 msi, nsis exe, dmg, deb, AppImage, zip 을 아티팩트로 업로드 | [build.yml L160-L176](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L160-L176) |

PDFium 을 빌드 중에 내려받지 않고 저장소에 커밋해 두는 이유는 워크플로 주석에 적혀 있다. 외부 배포처에서 받는 단계가 자주 실패했기 때문이다. Windows 는 `.msi` 외에 NSIS `.exe` 와 포터블 zip 도 만든다.

> **주의**: 태그 빌드는 `releaseDraft: true` 로 올라간다. 모든 플랫폼 빌드와 macOS 서명 검증이 끝난 뒤 사람이 draft 를 공개해야 사용자에게 보인다 ([build.yml L130-L132](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L130-L132)).

## 소스에서 빌드

README 의 소스 빌드 절차는 여섯 줄이다.

```bash
# 사전 요구: Node.js 20+, Rust 1.88+, protoc
git clone https://github.com/nashsu/llm_wiki.git
cd llm_wiki
npm install
npm --prefix mcp-server ci && npm run mcp:build
npm run tauri dev      # 개발
npm run tauri build    # 배포 번들
```

각 줄이 왜 필요한지는 `package.json`, `tauri.conf.json`, `Cargo.toml` 세 파일을 함께 봐야 이해된다.

### 툴체인 요구사항의 출처

| 요구 | 값 | 어디서 정해지나 |
|---|---|---|
| Rust | 1.88 이상 | [Cargo.toml L7-L10](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/Cargo.toml#L7-L10) 의 `rust-version = "1.88"`. 주석은 AnyDoc 과 그 압축 · 스프레드시트 의존성이 1.88 을 요구한다고 적는다 |
| Node.js | 20 | CI 와 릴리스 워크플로의 `node-version: 20` ([build.yml L98-L101](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L98-L101)) |
| protoc | 플랫폼 패키지 | 워크플로가 macOS 는 `brew install protobuf`, Ubuntu 는 `protobuf-compiler`, Windows 는 `arduino/setup-protoc` 로 설치한다. 어느 크레이트가 protoc 를 요구하는지는 코드에 적혀 있지 않다. LanceDB 의존성 트리일 가능성이 높다 (추정) |
| Linux 시스템 라이브러리 | webkit2gtk 4.1 등 | [build.yml L60-L69](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/.github/workflows/build.yml#L60-L69) 의 `libwebkit2gtk-4.1-dev libappindicator3-dev librsvg2-dev patchelf protobuf-compiler xdg-utils` |

README 는 Linux 시스템 라이브러리를 따로 적지 않는다. Linux 에서 소스 빌드를 할 때는 워크플로의 `apt-get install` 줄을 그대로 쓰면 된다.

### 코드 분석: npm scripts

빌드 명령이 서로를 어떻게 부르는지 본다. [package.json L6-L18](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/package.json#L6-L18)

```json
"scripts": {
  "dev": "vite",
  "typecheck": "tsc --build --pretty",
  "build": "npm run typecheck && vite build",
  "build:desktop": "npm --prefix mcp-server ci && npm run mcp:build && npm run build",
  "preview": "vite preview",
  "test": "npm run test:mocks && npm run test:llm",
  "test:mocks": "vitest run --exclude='**/*.real-llm.test.ts' --exclude='**/mcp-server/**'",
  "test:llm": "vitest run real-llm --no-file-parallelism --reporter=verbose",
  "mcp:build": "npm --prefix mcp-server run build",
  "mcp:test": "npm --prefix mcp-server test",
  "tauri": "tauri"
}
```

`tauri` 스크립트는 `@tauri-apps/cli` 를 그대로 부른다. 그래서 `npm run tauri dev` 와 `npm run tauri build` 는 Tauri CLI 의 `dev`, `build` 서브커맨드다. Tauri CLI 는 프론트엔드를 어떻게 띄우고 묶을지 `tauri.conf.json` 의 `build` 블록에서 읽는다.

| script | 역할 | 언제 쓰나 |
|---|---|---|
| `dev` | Vite 개발 서버 | `tauri dev` 가 `beforeDevCommand` 로 호출 |
| `typecheck` | `tsc --build` | `build` 의 첫 단계 |
| `build` | 타입 검사 후 `vite build` 로 `dist/` 생성 | 웹 번들만 필요할 때 |
| `build:desktop` | MCP 서버 설치 · 빌드 후 `build` | `tauri build` 가 `beforeBuildCommand` 로 호출 |
| `test:mocks` | 실제 LLM 을 부르지 않는 테스트 전체 | 평소 테스트 (14장) |
| `test:llm` | 파일명에 `real-llm` 이 든 테스트만 순차 실행 | API 키가 있을 때 (14장) |
| `mcp:build`, `mcp:test` | `mcp-server/` 하위 패키지 빌드 · 테스트 | MCP 서버 작업 (13장) |

### 코드 분석: tauri.conf.json 의 build 블록

`tauri dev` 와 `tauri build` 가 프론트엔드를 다루는 방식을 본다. [src-tauri/tauri.conf.json L6-L11](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/tauri.conf.json#L6-L11)

```json
"build": {
  "beforeDevCommand": "npm run dev",
  "devUrl": "http://localhost:1420",
  "beforeBuildCommand": "npm run build:desktop",
  "frontendDist": "../dist"
},
```

개발 모드에서는 Vite 서버를 `1420` 포트로 띄우고 webview 가 그 주소를 연다. Vite 설정도 같은 포트를 `strictPort: true` 로 고정한다 ([vite.config.ts L30-L32](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/vite.config.ts#L30-L32)). 1420 이 이미 쓰이고 있으면 다른 포트로 넘어가지 않고 실패한다.

배포 빌드에서는 `build:desktop` 이 MCP 서버까지 빌드한 다음 `dist/` 를 앱에 넣는다. 즉 `npm run tauri build` 만 실행해도 MCP 서버 빌드는 자동으로 따라온다. README 가 `mcp:build` 를 따로 실행하라고 하는 이유는 `tauri dev` 쪽에 있다. 개발 모드는 `npm run dev` 만 부르므로 MCP 서버가 빌드되지 않는다. 그런데 플랫폼별 설정 파일이 `mcp-server/dist` 를 번들 리소스로 지정한다.

```json
"resources": {
  "../mcp-server/package.json": "mcp-server/package.json",
  "../mcp-server/dist": "mcp-server/dist",
  "../mcp-server/node_modules": "mcp-server/node_modules"
}
```

위는 [src-tauri/tauri.macos.conf.json](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/tauri.macos.conf.json#L3-L7) 의 리소스 목록이다. Windows 와 Linux 설정 파일도 같은 세 항목에 각자의 PDFium 라이브러리(`pdfium.dll`, `libpdfium.so`)를 더한다. 리소스 경로가 없으면 Rust 빌드 스크립트 단계에서 오류가 나므로 개발 모드에서도 먼저 MCP 서버를 빌드해 두어야 한다 (추정, Tauri 빌드 스크립트의 리소스 검사 동작은 이 저장소 코드에서 확인하지 않았다).

### 창과 보안 설정

같은 `tauri.conf.json` 의 나머지 부분은 첫 실행 모습을 정한다.

| 키 | 값 | 의미 |
|---|---|---|
| `productName` / `identifier` | `LLM Wiki` / `com.llmwiki.app` | 앱 이름과 OS 식별자. 앱 데이터 폴더 이름이 identifier 를 따른다 |
| `version` | `0.6.13` | `package.json`, `Cargo.toml` 과 같은 값 |
| `windows[0]` | 1200 x 800, `titleBarStyle: Transparent` | 기본 창 크기. Windows 설정 파일은 `Visible` 로 덮어쓴다 |
| `security.csp` | `connect-src 'self' https: http:` 등 | webview 에서 임의 HTTP(S) 연결 허용 |
| `assetProtocol.scope` | `["**"]` | 로컬 이미지를 `asset:` 프로토콜로 표시 |
| `bundle.targets` | `all` | 플랫폼에서 가능한 번들 형식을 모두 생성 |

설정 전문은 [tauri.conf.json L1-L44](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/tauri.conf.json#L1-L44) 에 있다. `connect-src` 가 넓은 이유는 사용자가 지정하는 LLM 엔드포인트가 임의 주소이기 때문이다. 실제 LLM 요청은 대부분 Rust 쪽 HTTP 플러그인을 거치는데, 이 구조는 3장과 12장에서 다룬다.

### 릴리스 프로필

[Cargo.toml L77-L85](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/Cargo.toml#L77-L85) 의 `[profile.release]` 는 `codegen-units = 1`, `lto = true`, `opt-level = "s"`, `strip = true` 로 크기를 줄인다. 눈여겨볼 값은 `panic = "unwind"` 다. 주석은 서드파티 문서 파서가 잘못된 입력에서 panic 을 내더라도 Tauri command 경계에서 잡아 오류로 바꾸기 위해서라고 설명한다. 그 경계 처리(`panic_guard`)는 3장에서 본다.
