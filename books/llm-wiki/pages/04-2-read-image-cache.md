LLM 호출 전 단계에서 원본 텍스트를 추출하고 이미지를 처리하며, 변경 없는 원본은 증분 캐시로 건너뛴다.

## 원본 읽기: 텍스트 추출과 `.cache`

원본 텍스트는 Rust `read_file` 명령이 만든다. 확장자로 분기하고, 같은 폴더의 `.cache/<파일명>.txt` 가 원본보다 새로우면 캐시를 그대로 돌려준다.

| 확장자 | 처리 | 근거 |
|---|---|---|
| `pdf` | pdfium 으로 페이지별 텍스트, `## Page N` 머리글 | [fs.rs#L555-L591](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L555-L591) |
| `docx` · `pptx` · `xlsx` 등 Office 18종 | `anydoc` 으로 Markdown, 실패 시 호환 파서 | [fs.rs#L594-L629](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L594-L629) |
| `epub` · `mobi` | `ebook.rs` | [fs.rs#L95](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L95) |
| `org` | Org 문법을 Markdown 모양으로 변환 | [fs.rs#L240-L246](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L240-L246) |
| 이미지 · 미디어 | `[Image: 이름 (크기)]` 같은 자리표시 문자열 | [fs.rs#L96-L103](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L96-L103) |
| 그 밖 | UTF-8 텍스트로 읽기 | [fs.rs#L108-L122](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L108-L122) |

Office 캐시는 `.txt.parser` 표식 파일에 파서 버전(`anydoc-0.1.6-v1`)을 적어 두고, 버전이 다르면 캐시를 버린다([fs.rs#L33](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L33), [#L186-L203](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/fs.rs#L186-L203)). 무거운 추출은 `spawn_blocking` 으로 tokio 블로킹 풀에 넘긴다. 3장에서 다룬 Tauri command 구조가 여기서도 그대로 쓰인다.

### MinerU 기반 PDF 변환

PDF 이고 MinerU 가 켜져 있으면 내장 추출보다 먼저 MinerU 를 부른다. 핵심은 결과를 내장 추출과 같은 캐시 경로에 쓴다는 점이다.

코드 분석: MinerU 결과를 `.cache` 에 써서 뒤의 `read_file` 이 그대로 집어 가게 만드는 부분이다. [src/lib/ingest.ts#L742-L757](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L742-L757)

```ts
  const mineruConfigured = mineruCfg.backend === "local" || Boolean(mineruCfg.token)
  if (isPdf && mineruCfg.enabled && mineruConfigured) {
    try {
      const cacheDir = sp.substring(0, sp.lastIndexOf("/"))
      const cachePath = `${cacheDir}/.cache/${fileName}.txt`
      activity.updateItem(activityId, { detail: "MinerU: parsing PDF..." })
      console.log(`[ingest:mineru] submitting "${fileName}" to MinerU API`)
      const mineruResult = await parseWithMineruResult(mineruCfg, sp, undefined, (msg) => {
        activity.updateItem(activityId, { detail: `MinerU: ${msg}` })
      }, signal, {
        projectPath: pp,
        sourceSummarySlug,
      })
      await createDirectory(`${cacheDir}/.cache`)
      await writeFile(cachePath, mineruResult.markdown)
      mineruSavedImages = mineruResult.savedImages
```

MinerU 가 실패하면 경고만 남기고 내장 pdfium 추출로 넘어간다. 캐시 파일 수정 시각이 원본보다 새로워지므로 Rust `read_cache` 가 이후 호출에서 MinerU 결과를 돌려준다. 별도의 "어느 파서를 썼는가" 상태 없이 파일 시각만으로 두 파서를 전환하는 구조다.

`mineru.ts` 의 주요 상수와 동작은 아래와 같다([mineru.ts#L8-L13](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/mineru.ts#L8-L13)).

| 항목 | 값 | 비고 |
|---|---|---|
| 클라우드 API | `https://mineru.net/api/v4` | 토큰 필요, 모델은 `pipeline` 또는 `vlm` |
| 로컬 기본 엔드포인트 | `http://127.0.0.1:8000` | `/health` 로 V1 API 여부를 판별 |
| 폴링 간격 | 3,000 ms | |
| 클라우드 폴링 한도 | 300,000 ms (5분) | |
| 로컬 폴링 한도 | 3,600,000 ms (1시간) | 주석: 모델 cold start 포함 |
| 파일 크기 한도 | 200 MB | 클라우드 · 로컬 공통 |

결과 zip 에서는 `full.md` 를 우선 고르고, HTML 표를 Markdown 표로 바꾼다. zip 안의 이미지는 `wiki/media/<slug>/mineru/` 에 저장하고 Markdown 안의 이미지 경로를 그 위치로 다시 쓴다([mineru.ts#L584-L634](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/mineru.ts#L584-L634), [#L649-L695](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/mineru.ts#L649-L695)). 이미지 저장이 실패해도 텍스트는 살린다.

## 이미지: 추출, 캡션, 안전망

이미지 처리는 세 단계다. 원본에서 이미지를 꺼내 `wiki/media/<slug>/` 에 저장하고, 비전 모델로 캡션을 붙이고, 생성이 끝난 뒤 요약 페이지 끝에 이미지 절을 붙인다.

### 추출

`extractAndSaveSourceImages` 는 확장자로 분기한다. PDF 는 Rust `extract_and_save_pdf_images_cmd`, `pptx` · `docx` 는 `extract_and_save_office_images_cmd` 를 부르고, 원본 자체가 `png` · `jpg` · `jpeg` 이면 복사만 한다([extract-source-images.ts#L152-L216](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/extract-source-images.ts#L152-L216)). Markdown 원본 안의 로컬 이미지 참조는 `extractAndSaveMarkdownImages` 가 같은 폴더로 복사한다. PDF 에 MinerU 이미지가 이미 있으면 pdfium 이미지 추출은 건너뛴다.

코드 분석: Rust 추출기의 거름 기준이다. [src-tauri/src/commands/extract_images.rs#L60-L71](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/extract_images.rs#L60-L71)

```rust
impl Default for ExtractOptions {
    fn default() -> Self {
        Self {
            min_width: 100,
            min_height: 100,
            max_images: 500,
            skip_full_page_images: true,
            full_page_coverage_ratio: 0.90,
            full_page_min_text_chars: 80,
        }
    }
}
```

| 필드 | 기본값 | 의미 |
|---|---|---|
| `min_width` · `min_height` | 100 · 100 | 어느 한 변이라도 작으면 아이콘 · 로고로 보고 버림 |
| `max_images` | 500 | 문서당 상한. 메모리와 이후 캡션 비용을 막는 장치 |
| `skip_full_page_images` | `true` | 페이지 대부분을 덮는 래스터를 버릴지 |
| `full_page_coverage_ratio` | 0.90 | 가로 · 세로 모두 이 비율 이상 덮으면 "페이지 크기" (0.5~1.0 으로 고정 범위 처리) |
| `full_page_min_text_chars` | 80 | 이 이상 텍스트가 있는 페이지에서만 페이지 크기 이미지를 버림 |

마지막 두 필드는 스캔 PDF 를 위한 것이다. 텍스트 층이 없는 스캔 페이지는 이미지가 유일한 내용이므로 남기고, 텍스트가 충분한 페이지의 배경 래스터는 중복이라 버린다. 이미지는 모두 PNG 로 다시 인코딩해 `img-<번호>.png` 로 저장하고 SHA-256 을 함께 돌려준다([#L836-L880](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src-tauri/src/commands/extract_images.rs#L836-L880)).

### 캡션

추출한 이미지는 원본 텍스트 끝에 `## Referenced Local Images` 절로 덧붙여 캡션 파이프라인에 넘긴다([ingest.ts#L69-L77](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L69-L77)). 캡션이 필요한 이유는 주석에 적혀 있다. alt 텍스트가 빈 이미지는 요약 과정에서 LLM 이 지워 버리기 쉽고, 캡션이 붙으면 생성 단계에서 이미지 참조가 살아남을 가능성이 높아진다.

코드 분석: 비전 모델에 보내는 고정 프롬프트다. [src/lib/vision-caption.ts#L70-L71](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/vision-caption.ts#L70-L71)

```ts
export const CAPTION_PROMPT =
  "Describe this image factually for a knowledge-base index. Include: any visible text verbatim, chart axes and values, diagram structure (boxes/arrows/labels), key visual elements. Do NOT speculate or editorialize. 2 to 4 sentences. Output plain text only — no markdown, no preamble."
```

이미지 앞뒤 150자(`CONTEXT_CHARS`)를 주변 문맥으로 함께 보내고, 문맥이 그림을 가리키면 거기에 맞추라는 지시가 추가된다([image-caption-pipeline.ts#L228](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/image-caption-pipeline.ts#L228)). 캡션 캐시는 `.llm-wiki/image-caption-cache.json` 이고 키는 이미지 바이트의 SHA-256 에 출력 언어를 붙인 것이다([#L116-L119](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/image-caption-pipeline.ts#L116-L119), [#L518-L525](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/image-caption-pipeline.ts#L518-L525)). 여러 문서에 반복되는 로고나 차트 틀은 한 번만 캡션한다. 동시성은 사용자 설정값이고, 캐시는 배치가 끝난 뒤 한 번만 기록한다. Codex CLI 전송은 이미지 입력을 지원하지 않아 캡션을 건너뛴다([#L444-L454](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/image-caption-pipeline.ts#L444-L454)).

멀티모달 설정이 꺼져 있으면 캡션만 건너뛰는 게 아니라 원본 텍스트에서 이미지 참조를 지운다([ingest.ts#L989-L995](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L989-L995)). 이미지 파일은 디스크에 남지만 위키 페이지와 검색 색인에는 들어가지 않는다.

### 안전망 절

생성이 끝나면 `injectImagesIntoSourceSummary` 가 요약 페이지 끝에 `<!-- llm-wiki:embedded-images -->` 표식 한 쌍으로 감싼 `## Embedded Images` 절을 붙인다([ingest.ts#L3514-L3598](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L3514-L3598)). 이미지는 페이지 번호별로 묶이고 alt 에는 캐시의 캡션이 들어간다. 다시 수집할 때는 표식 사이를 지우고 새로 쓰므로 여러 번 돌려도 절이 쌓이지 않는다. 요약 페이지가 없으면 최소한의 frontmatter 를 가진 대체 페이지를 만든다. 이미지가 참조 없이 `wiki/media/` 에만 남으면 고아 정리 과정에서 지워질 수 있기 때문이다.

## 증분 캐시: 변경 없는 원본 생략

원본을 읽은 직후 `checkIngestCache` 가 `.llm-wiki/ingest-cache.json` 을 본다. 키는 원본 식별자(`raw/sources/` 아래 상대 경로, 5장)이고 값은 내용의 SHA-256 과 그때 쓴 파일 목록이다.

코드 분석: 캐시 적중의 조건이다. [src/lib/ingest-cache.ts#L67-L99](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest-cache.ts#L67-L99)

```ts
  const cache = await loadCache(projectPath)
  const entry = cache.entries[sourceFileName]
  if (!entry) return null

  const currentHash = await sha256(sourceContent)
  if (entry.hash !== currentHash) return null

  const pp = normalizePath(projectPath)
  for (const filePath of entry.filesWritten) {
    const fullPath = isAbsolutePath(filePath)
      ? normalizePath(filePath)
      : `${pp}/${filePath}`
    try {
      if (!(await fileExists(fullPath))) {
        // ...
        return null
      }
```

해시가 같아도 그때 쓴 파일이 하나라도 사라졌으면 캐시를 무시한다. 해시 대상이 원본 바이트가 아니라 추출된 텍스트라는 점도 눈여겨볼 만하다. MinerU 를 새로 켜서 추출 결과가 바뀌면 같은 PDF 라도 다시 수집된다.

캐시가 적중해도 이미지 단계는 다시 돈다. 이전 버전 앱으로 수집해 이미지가 없던 원본도 이번 파이프라인 계약에 맞춰지게 하려는 것이고, 경로가 결정적이며 표식으로 감싼 교체라서 반복해도 결과가 같다([ingest.ts#L805-L815](https://github.com/nashsu/llm_wiki/blob/e08cfce65aad8b76b13546a081c1a8d4422b70c3/src/lib/ingest.ts#L805-L815)).
